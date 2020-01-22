import collections
import numpy as np
import pickle


class ReplayMemory:
    def __init__(self, maxlen=None):
        self.buffer = collections.deque(maxlen=maxlen)
    
    def append(self, experience):
        self.buffer.append(experience)
        
    def sample(self, sample_size):
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        observations, actions, rewards, dones, next_observations = zip(*[self.buffer[i] for i in indices])
        return np.asarray(observations), np.asarray(actions), np.asarray(rewards), np.asarray(dones), np.asarray(next_observations)

    def __len__(self):
        return len(self.buffer)

    @staticmethod
    def from_replays(replays):
        memory = ReplayMemory()
        if isinstance(replays, Replay):
            replays = [replays]

        for replay in replays:
            if len(replay) <= 0:
                continue
            action, observation, reward, done, info = replay[0]
            for i in range(1, len(replay)):
                action, next_observation, reward, done, info = replay[i]
                memory.append((observation, action, reward, done, next_observation))
                observation = next_observation

        return memory


class Replay:
    def __init__(self):
        self.replay = []

    def append(self, action, observation, reward, done, info):
        self.replay.append((action, observation, reward, done, info))

    def __len__(self):
        return len(self.replay)

    def __getitem__(self, item):
        return self.replay[item]

    def __setitem__(self, key, value):
        self.replay[key] = value

    def __iter__(self):
        yield from self.replay

    def dump(self, file):
        pickle.dump(self, file)

    @staticmethod
    def load(file):
        return pickle.load(file)
