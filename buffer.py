import numpy as np
import queue


class ReplayBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(buffer_size)

    def add(self, state, action, next_state, reward, done):
        if not self.buffer.full():
            self.buffer.put([state, action, next_state, reward, done])

    def sample(self, batch_size):
        contents = list()
        for _ in batch_size:
            if not self.buffer.empty():
                contents.append(self.buffer.pop())
        return contents