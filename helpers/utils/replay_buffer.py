import numpy as np

class ReplayBuffer():
    def _init_(self, max_size=10000, batch_size=64, seed=0):
        # initialize seed
        self.rand_generator = np.random.RandomState(training_info["seed"])

