import numpy as np
import torch


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



def collate_buffer(buffer, batch_size):
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    for state, action, next_state, reward, done in buffer.sample(batch_size):
        states.append(torch.tensor(state, dtype=torch.float32))
        actions.append(torch.tensor(action, dtype=torch.float32))
        next_states.append(torch.tensor(next_state, dtype=torch.float32))
        rewards.append(torch.tensor([reward], dtype=torch.float32))
        dones.append(torch.tensor([done], dtype=torch.bool))
        # states.append(state)
        # actions.append(action)
        # next_states.append(next_state)
        # rewards.append(reward)
        # dones.append(done)

    states = torch.stack(states, dim=0)
    actions = torch.stack(actions, dim=0)
    next_states = torch.stack(next_states, dim=0)
    rewards = torch.stack(rewards, dim=0)
    dones = torch.stack(dones, dim=0)

    return states, actions, next_states, rewards, dones