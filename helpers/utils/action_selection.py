# Adapted from https://github.com/mimoralea/gdrl
import torch
import numpy as np
import sys

class GreedyStrategy():
	def __init__(self, bounds):
		# bounds can be a value or an array depending on number of actions
		self.low, self.high = bounds
		self.ratio_noise_injected = 0

	def select_action(self, model, state):
		with torch.no_grad():
			greedy_action = model(state).cpu().detach().data.numpy().squeeze()
		
		action = np.clip(greedy_action, self.low, self.high)
		return np.reshape(action, self.high.shape)

class NormalNoiseStrategy(object):
	def __init__(self, bounds, exploration_noise_ratio=0.1, seed=0):
		super(NormalNoiseStrategy, self).__init__()
		self.low, self.high = bounds
		self.exploration_noise_ratio = exploration_noise_ratio
		self.ratio_noise_injected = 0
		self.rand_generator = np.random.RandomState(seed)

	def select_action(self, model, state, max_exploration=False):
		if max_exploration:
			noise_scale = self.high
		else:
			noise_scale = self.exploration_noise_ratio * self.high

		with torch.no_grad():
			greedy_action = model(state).cpu().detach().data.numpy().squeeze()
		# creating and adding gaussian noise to action
		noise = self.rand_generator.normal(loc=0, scale=noise_scale, size=len(self.high))
		noisy_action = greedy_action + noise
		action = np.clip(noisy_action, self.low, self.high)	
		# calculate ratio of added noise
		self.ratio_noise_injected = np.mean(abs((greedy_action - action) / (self.high - self.low)))	
		return action