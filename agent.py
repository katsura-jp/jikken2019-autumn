import numpy as np
import pickle
import gym

class Agent(object):
    def save_models(self, path):
        pass
    def load_models(self, path):
        pass
    def select_action(self, state):
        raise NotImplementedError
    def select_exploratory_action(self, state):
        raise NotImplementedError
    def train(self, state, action, next_state, reward, done):
        pass


class RandomAgent(Agent):
    def __init__(self,action_size, action_low=-2., action_high=2., eval_seed=5):
        self.action_low = action_low
        self.action_high = action_high
        self.action_size = action_size
        self.eval_seed = 5

        self.state = None

    def save_models(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.state, f)

    def load_models(self, path):
        with open(path, 'rb') as f:
            self.state = pickle.load(f)

    def select_action(self, state):
        raise (self.action_high + self.action_low) / 2

    def select_exploratory_action(self, state):
        return np.random.uniform(self.action_low, self.action_high, self.action_size)

    def eval(self, env, n_episode=10, n_step=1000, seed=0, mean=True):
        rewards = []

        # env.seed(seed)

        for e in range(n_episode):
            state = env.reset()
            reward_sum = 0.
            for i in range(n_step):
                action = self.select_exploratory_action(state)
                next_state, reward, done, info = env.step(action)
                reward_sum += reward
                state = next_state
                if done:
                    break
            rewards.append(reward_sum)
        env.close()
        rewards = np.array(rewards)
        if mean:
            return rewards.mean()
        else:
            return rewards



class TQLAgent(Agent):
    def __init__(self, state_dim, action_dim, state_size=10, action_size=9,
                 lr=3e-4, gamma=0.99, eps=0.05, batch_size=256, max_iter=5e+5,
                 state_low=-2, state_high=2, action_low=-2, action_high=2):
        '''
        :param state_dim: 状態の出力次元数
        :param action_dim: 行動の出力次元数
        :param state_size: 状態の出力次元数
        :param action_size: 行動の出力次元数
        :param lr: 学習率
        :param gamma: 割引報酬率
        '''

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.batch_size = batch_size
        self.max_iter = max_iter

        # 状態空間
        self.state_space = np.random.normal(state_low,state_high,(state_size, state_dim))
        # 行動空間
        self.action_space = np.random.normal(action_low,action_high,(action_size, action_dim))

        self.q_table = np.random.normal(0,1,(state_size, action_size)) * 1e-8
        # Qテーブル Q[idx(s), idx(a)]

        self._elapsed_steps = 0

    def save_models(self, path):
        # Qテーブルを保存する。
        with open(path, 'wb') as f:
            pickle.dump(self.q_table)

    def load_mdoels(self, path):
        # Qテーブルを読み込む。
        with open(path, 'rb') as f:
            q_table = pickle.load(f)
        assert q_table.shape == self.q_table.shape
        self.q_table = q_table

    def select_action(self, state):
        # actionを返す。
        idx = np.argmax([self.q_table[state, j] for j in self.action_size])
        return self.lr * self.action_space[idx]

    def select_exploratory_action(self, state):
        # 行動方策からアクションをサンプルする。 $\alpha \textasciitilde \beta(\alpha ; \pi(state))$
        if np.random.rand() > self.eps:
            return self.select_action(state)
        else:
            idx = np.random.choice(self.action_size)
            return self.action_space[idx]


    def train(self, state, action, next_state, reward, done):
        if done:
            pass
        else:
            omega = reward + self.gamma * np.max([self.q_table[next_state, j] for j in self.action_size]) - self.q_table[state, action]
            self.q_table[state, action] = self.q_table[state, action] + self.lr * omega
            pass
