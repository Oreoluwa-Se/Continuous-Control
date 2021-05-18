# Adapted from https://github.com/mimoralea/gdrl
from helpers.utils.action_selection import GreedyStrategy, NormalNoiseStrategy
from helpers.utils.priority_replay import Memory
from helpers.nn.network import FCQV, FCDP
from itertools import count
import torch.optim as optim
import numpy as np
import torch
import time
import glob
import os
import gc

LEAVE_PRINT_EVERY_N_SECS = 300
RESULTS_DIR = os.path.join('..', 'results')
ERASE_LINE = '\x1b[2K'

class DDPG_agent:
    def __init__(self, policy_info={}, value_info={}, env_info={}, training_info={}, buffer_info={}):
        # set device target
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ### POLICY NETWORK PARAMETERS
        self.online_policy_model = FCDP(env_info["state_size"], env_info["bounds"], 
                                    policy_info["hidden_dims"], device=self.device)
        self.target_policy_model = FCDP(env_info["state_size"], env_info["bounds"], 
                                    policy_info["hidden_dims"], device=self.device)
        self.policy_optimizer = optim.Adam(self.online_policy_model.parameters(), lr=policy_info["learning_rate"])
        self.policy_max_grad_norm = policy_info["max_grad_norm"]

        ### VALUE NETWORK PARAMETERS
        self.online_value_model = FCQV(env_info["state_size"], env_info["action"], 
                                    value_info["hidden_dims"], device=self.device)
        self.target_value_model = FCQV(env_info["state_size"], env_info["action"], 
                                    value_info["hidden_dims"], device=self.device)
        self.value_optimizer = optim.Adam(self.online_value_model.parameters(), lr=value_info["learning_rate"])
        self.value_max_grad_norm = value_info["max_grad_norm"]

        # TRAINING AND EVALUATION STRATEGY
        self.training_strategy = NormalNoiseStrategy(env_info["bounds"], training_info["exploration_noise_ratio"])
        self.update_target_every_steps  = training_info["update_every_step"]
        self.evaluation_strategy = GreedyStrategy(env_info["bounds"])
        self.n_warmup_batches = training_info["n_warmup_batches"]
        self.soft_weight_tau = training_info.get("weight_mix_ratio", None)  

        # MEMORY INITIALIZATION
        self.replay_buffer = Memory(capacity=buffer_info["size"], seed=training_info["seed"])
        self.batch_size = buffer_info["batch_size"]

        # seed
        torch.manual_seed(training_info["seed"]);
        self.rand_generator = np.random.RandomState(training_info["seed"])

        # lower and upper action value bounds
        self.low_bounds, self.high_bounds = env_info["bounds"] 

    def prepopulate(self, brain_name, env):
        """
            First thing called after environment has been setup
            To aviod the empty memory problem we randomly pre populate the memory. This is done
            by taking random actions and storing them as experiences
            Args:
                brain_name: (string) name of agent we are using
                env: (object) Environment we are operating in
        """
        # flag for when to reset the environment [when we hit a terminal state]
        reset_check, last_state = False, None
        
        for idx in range(self.replay_buffer.tree.capacity):
            # if idx is the first step get state or we have to reset
            if idx == 0 or reset_check:
                # change reset check back to false
                reset_check = False
                # resent environment and extract current state
                env_info   = env.reset(train_mode=True)[brain_name]
                last_state = env_info.vector_observations[0]

            # take random actions within acceptable bounds
            action = self.rand_generator.uniform(low=self.low_bounds,
                                                 high=self.high_bounds,
                                                 size=len(self.high_bounds))
            # take the action, recod reward, and terminal status
            env_info = env.step(action)[brain_name]
            reward   = env_info.rewards[0]
            done     = env_info.local_done[0]

            # checking status
            if done:
                # set reset flag
                reset_check = True
                state = np.zeros(last_state.shape)
                # store in replay
                self.replay_buffer.store(last_state, action, reward, state, done)
            else:
                # get next state from the environment
                state = env_info.vector_observations[0]
                # store in replay
                self.replay_buffer.store(last_state, action, reward, state, done)
                # update state
                last_state = state
        
    def update_networks(self, tau=0.1):
        tau = self.soft_weight_tau if self.soft_weight_tau is not None else tau
        # copy value model
        for target, online in zip(self.target_value_model.parameters(), 
                                  self.online_value_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        # copy policy model
        for target, online in zip(self.target_policy_model.parameters(), 
                                  self.online_policy_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def load(self, states, actions, rewards, next_states, is_terminals):        
        # convert to torch tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)

        # returns tensors
        return states, actions, rewards, next_states, is_terminals

    def optimize_model(self):
        # priotized replay used so we can update optimize on the go: we get the batch indexes, memory, 
        # importance sampling
        idx_batch, memory_batch, ISweights = self.replay_buffer.sample_per(self.batch_size)

        # convert sampling weights to tensor
        ISweights = torch.from_numpy(ISweights).float().to(self.device)
        # unwrap
        states, actions, rewards, next_states, is_terminals = self.replay_buffer.unwrap_experiences(memory_batch)
        # convert to torch
        states, actions, rewards, next_states, is_terminals = self.load(states, actions, rewards, next_states, is_terminals)
        
        ## Target policy 
        # get maximum policy over all states
        argmax_a_q_sp = self.target_policy_model(next_states)
        # calculate the q values corresponding to the policy above
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        # calculate the TD target q_state action values
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        ## Online value
        # for each state action pair we calculate the q_values
        q_sa = self.online_value_model(states, actions)

        ## Loss calculations
        td_error_loss = q_sa - target_q_sa.detach()
        # calculate absolute error: convert to numpy
        abs_error = torch.abs(td_error_loss).cpu().detach().numpy()
        # update PER 
        self.replay_buffer.batch_update(idx_batch, abs_error.squeeze())
        # calculate value loss using weight mean square error
        value_loss = (ISweights * td_error_loss).mul(0.5).pow(2).mean()

        # zero optimizer, do a backward pass, clip gradients, and finally optimizer
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), 
                                        self.value_max_grad_norm)
        self.value_optimizer.step()

        ## ONLINE POLICY
        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)

        ## calculate loss using weighted mean
        policy_loss = -(ISweights * max_a_q_s).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(),
                                        self.policy_max_grad_norm)
        self.policy_optimizer.step()

    def interaction_step(self, last_state, env, brain_name):
        # initially we randomly explore the sample space
        check = self.replay_buffer.current_storage_size <  self.n_warmup_batches * self.batch_size        
        action = self.training_strategy.select_action(self.online_policy_model, last_state, check)
        
        # get environment values
        env_info  = env.step(action)[brain_name]
        state     = env_info.vector_observations[0]
        reward    = env_info.rewards[0]
        done      = env_info.local_done[0]
        
        # store into replay buffer
        self.replay_buffer.store(last_state, action, reward, state, done)
        
        # update tracking parameters
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += self.training_strategy.ratio_noise_injected

        # return values
        return state, done

    def train(self, env, brain_name, gamma, max_minutes, max_episodes, goal_mean_100_reward):
        # initialize tracking parameters
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []
        self.gamma = gamma

        # loop parameters 
        total_steps = 0
        training_time = 0
        training_start, last_debug_time = time.time(), float("-inf")

        # storage for results
        results = np.empty((max_episodes, 5))
        results[:] = np.nan

        # start training loop
        for episode in range(1, max_episodes + 1):
            # episode start tick
            episode_start = time.time()

            # refresh environment
            state = env.reset(train_mode=True)[brain_name].vector_observations[0]
            is_terminal =  False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env, brain_name)
                # after making random steps
                check = self.replay_buffer.current_storage_size >  (self.n_warmup_batches * self.batch_size)
                if check:
                    # run optimization
                    self.optimize_model()
                
                # updates every episode
                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_networks()

                if is_terminal:
                    gc.collect()
                    break

            # stat tracking
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_policy_model, env, brain_name)
            self.save_checkpoint(episode - 1, self.online_policy_model)

            total_steps = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            # mean and std calculations
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])

            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])

            lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            results[episode - 1] = total_steps, mean_100_reward, mean_100_eval_score, \
                                training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            # termination criteria check
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

            # message string
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:07}, '
            debug_message += 'ar_10 ts {:05.1f} \u00B1 {:05.1f}, '
            debug_message += 'ar_100 ts {:05.1f} \u00B1 {:05.1f}, '
            debug_message += 'ex 100 {:02.1f} \u00B1 {:02.1f}, '
            debug_message += 'ev {:05.1f} \u00B1 {:05.1f}'
            debug_message = debug_message.format(elapsed_str, episode - 1, total_steps,
                                                 mean_10_reward, std_10_reward, 
                                                 mean_100_reward, std_100_reward,
                                                 mean_100_exp_rat, std_100_exp_rat,
                                                 mean_100_eval_score, std_100_eval_score)
            print(debug_message, flush=True)
            if reached_debug_time or training_over:
                print("Debug Message")
                print(debug_message, flush=True)
                last_debug_time = time.time()

            if training_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break

        # get score for last round
        final_eval_score, score_std = self.evaluate(self.online_policy_model, env, brain_name, n_episodes=100)
        wallclock_time = time.time() - training_start
        print("Training complete.")
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))

        # close and delete the environment
        env.close() ; del env
        self.get_cleaned_checkpoints()
        return results, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, brain_name, n_episodes=1):
        rs = []

        for _ in range(n_episodes):
            env_info   = eval_env.reset(train_mode=True)[brain_name]
            state, done = env_info.vector_observations[0], False
            rs.append(0)
            for _ in count():
                action = self.evaluation_strategy.select_action(eval_policy_model, state)
                env_info = eval_env.step(action)[brain_name]
                state    = env_info.vector_observations[0]
                reward   = env_info.rewards[0]
                done     = env_info.local_done[0]
                rs[-1] += reward

                if done: break

        # return mean and standard deviation
        return np.mean(rs), np.std(rs)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join("results", "checkpoint_models", 'model.{}.tar'.format(episode_idx)))

    def get_cleaned_checkpoints(self, n_checkpoints=4):
        try:
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join("results", "checkpoint_models", '*.tar'))
        paths_dic = {int(path.split('.')[-2]): path for path in paths}
        last_ep = max(paths_dic.keys())
        checkpoint_idxs = np.linspace(1, last_ep + 1, n_checkpoints, endpoint=True, dtype=np.int) - 1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

     













    
        

        

