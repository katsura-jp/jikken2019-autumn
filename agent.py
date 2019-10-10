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
    def __init__(self,action_space, observation_space):

        self.action_space = action_space
        self.observation_space = observation_space

        self.state = None

    def save_models(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.state, f)

    def load_models(self, path):
        with open(path, 'rb') as f:
            self.state = pickle.load(f)

    def select_action(self, state):
        raise (self.action_space.high + self.action_space.low) / 2

    def select_exploratory_action(self, state):
        return np.random.uniform(self.action_space.low, self.action_space.high, self.action_space.shape)

    def eval(self, env, n_episode=10, n_step=1000, seed=5, mean=True):
        # 4.1.5 エージェント評価用関数の実装
        rewards = [] # 各エピソードの累積報酬を格納する
        env.seed(seed)
        for e in range(n_episode):
            state = env.reset()
            reward_sum = 0. # 累積報酬
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
    def __init__(self, action_space, observation_space, state_size=10, action_size=9,
                 lr=3e-4, gamma=0.99, eps=0.05, batch_size=256, max_iter=5e+5):
        '''
        :param action_space: 行動空間
        :param observation_space: 状態空間
        :param state_size: 行動空間の分割数
        :param action_size: 状態空間の分割数
        :param lr: 学習率
        :param gamma: 割引報酬率
        :param eps: eps-greedy法における確率
        :param batch_size: バッチサイズ
        :param max_iter: 最大ステップ数
        '''

        self.action_space = action_space
        self.observation_space = observation_space

        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.batch_size = batch_size
        self.max_iter = max_iter

        self.q_table = np.random.normal(0,1,(state_size**self.observation_space.shape[0], action_size**self.action_space.shape[0])) * 1e-8
        # Qテーブル Q[idx(s), idx(a)]


        self._elapsed_steps = 0

        #TODO: index返すためのベクトルを作る
        # np.linspaceでベクトル列を作成
        # np.searchsorted でindexを返す
        # divmod(np.searchsorted(a.ravel(), target), a.shape[1])[0]
        self._action_index_ref = np.linspace(self.action_space.low, self.action_space.high, self.action_size).T
        self._state_index_split = np.linspace(self.observation_space.low, self.observation_space.high, self.state_size).T


    def _action_index(self, action):
        '''
        行動のインデックスを返す
        '''

        pass

    def _state_index(self, state):
        '''
        状態のインデックスを返す
        '''
        pass


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
        # ターゲット方策 π
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
