import random
import numpy as np
import queue


class ReplayBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.buffer = list()

    def add(self, state, action, next_state, reward, done):
        if self.buffer_size > len(self.buffer):
            self.buffer.append([state, action, next_state, reward, done])
        else:
            # FIFO(First In, First Out)
            del self.buffer[0]
            self.buffer.append([state, action, next_state, reward, done])

    def sample(self, batch_size):
        indice = np.random.randint(len(self.buffer), size=batch_size)
        contents = list()
        for idx in indice:
            contents.append(self.buffer[idx])
        return iter(contents)

    def __len__(self):
        return len(self.buffer)

