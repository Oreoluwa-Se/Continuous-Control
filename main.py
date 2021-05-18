from unityagents import UnityEnvironment
from helpers.agent import DDPG_agent
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys


def environment_start():
    path = os.path.join("Reacher.exe")
    env = UnityEnvironment(file_name=path)
    # get default brain
    brain_name = env.brain_names[0] 
    # return environment, brain_name
    return env, brain_name

def plot(results):
    # total_steps, mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed
    fig, axs = plt.subplots(2, 1, figsize=(15,10), sharey=False, sharex=True)
    axs[1].plot(results[:,1], 'r', label="mean 100 training_score", linewidth=2)
    axs[0].plot(results[:,2], 'r', label="mean 100 evaluation score", linewidth=2)    

    # Title
    axs[0].set_title("Average Reward (Evaluation)")
    axs[1].set_title("Average Reward (Training)")

    plt.xlabel("Episodes")
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")

    plt.savefig(os.path.join('results','scores_plot.png'))
    plt.show()


if __name__ == "__main__":
    env, brain_name = environment_start()
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    
    # information about the environment
    env_info = {"state_size": env_info.vector_observations.shape[1],
                "action": brain.vector_action_space_size,
                "bounds": (np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1])),
                "gamma": 0.95,
                "max_minutes": 180,
                "max_episodes": 500,
                "goal_mean_100_reward": 30}
    
    # information for policy model
    policy_info = {"hidden_dims":[256, 256],
                   "learning_rate":0.0005,
                   "max_grad_norm":float("inf")}
    # information for value model
    value_info = {"hidden_dims":[256, 256],
                   "learning_rate":0.0005,
                   "max_grad_norm":float("inf")}
    # general training information
    training_info = {"exploration_noise_ratio": 0.1,
                     "update_every_step": 1,
                     "n_warmup_batches": 5,
                     "weight_mix_ratio": 0.001,
                     "seed":0}
    # buffer_information
    buffer_info = {"size":100000,
                   "batch_size": 256}
    # load agent
    agent = DDPG_agent(policy_info, value_info, env_info, training_info, buffer_info)
    # populate - 
    agent.prepopulate(brain_name, env)
    # get final over episodes
    results, final_eval_score, training_time, wallclock_time = agent.train(env, brain_name, 
                                                                          env_info["gamma"], 
                                                                          env_info["max_minutes"],
                                                                          env_info["max_episodes"], 
                                                                          env_info["goal_mean_100_reward"])
    # plot results
    plot(results)
    
    



