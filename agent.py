import numpy as np
import pickle
import gym

class Agent(object):
    def save_models(self, path):
        pass
    def load_models(self, path):
        pass
    def select_action(self, state):
        pass
    def select_exploratory_action(self, state):
        pass
    def train(self, state, action, next_state, reward, done):
        pass


class RandomAgent(Agent):
    '''ランダムエージェント'''
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
        return (self.action_space.high + self.action_space.low) / 2

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
                # action = self.select_action(state)
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
    '''
    テーブルQ学習のエージェント.
    '''
    def __init__(self, action_space, observation_space, state_size=10, action_size=9,
                 lr=3e-4, gamma=0.99, eps=0.05):
        '''
        :param action_space: 行動空間
        :param observation_space: 状態空間
        :param state_size: 行動空間の分割数
        :param action_size: 状態空間の分割数
        :param lr: 学習率
        :param gamma: 減衰率
        :param eps: eps-greedy法における確率
        '''

        self.action_space = action_space
        self.observation_space = observation_space

        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr
        self.gamma = gamma
        self.eps = eps

        self.q_table = np.random.normal(0, 1,(state_size**self.observation_space.shape[0], action_size**self.action_space.shape[0])) * 1e-8
        # Qテーブル Q[idx(s), idx(a)]

        # indexの参照
        # action_index_refはactionの部分空間でもある

        action_margin = (self.action_space.high - self.action_space.low) / (self.action_size*2)
        state_margin = (self.observation_space.high - self.observation_space.low) / (self.state_size*2)
        self._action_index_ref = np.linspace(self.action_space.low + action_margin, self.action_space.high - action_margin, self.action_size)
        self._state_index_ref  = np.linspace(self.observation_space.low + state_margin, self.observation_space.high - state_margin, self.state_size)
        self._action_power = np.power(self.action_size, np.arange(self.action_space.shape[0]))
        self._state_power = np.power(self.state_size, np.arange(self.observation_space.shape[0]))


    def _action_index(self, action):
        '''行動のインデックスを返す
        代表点にもっとも近いindexを返す'''
        index = (np.argmin(np.abs(self._action_index_ref - action), axis=0) * self._action_power).sum()
        return index

    def _state_index(self, state):
        '''状態のインデックスを返す'''
        index = (np.argmin(np.abs(self._state_index_ref - state), axis=0) * self._state_power).sum()
        return index

    def _init_q_table(self):
        '''Qテーブルの初期化'''
        self.q_table = np.random.normal(0,1,(self.state_size**self.observation_space.shape[0], self.action_size**self.action_space.shape[0])) * 1e-8

    def save_models(self, path):
        '''Qテーブルを保存する。'''
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_models(self, path):
        # Qテーブルを読み込む。
        with open(path, 'rb') as f:
            q_table = pickle.load(f)
        assert q_table.shape == self.q_table.shape
        self.q_table = q_table

    def select_action(self, state):
        '''ターゲット方策πを返す'''
        state_index = self._state_index(state)
        index = np.argmax([self.q_table[state_index, j] for j in range(self.q_table.shape[1])])
        return self._action_index_ref[index]

    def select_exploratory_action(self, state):
        # 行動方策からアクションをサンプルする。 $\alpha \textasciitilde \beta(\alpha ; \pi(state))$
        if np.random.rand() > self.eps:
            return self.select_action(state)
        else:
            index = np.random.choice(self.q_table.shape[1])
            return self._action_index_ref[index]

    def train(self, state, action, next_state, reward, done):
        # doneは特に使わない。

        # actionとstateのindexを取得する。
        next_state_index = self._state_index(next_state)
        state_index = self._state_index(state)
        action_index = self._action_index(action)

        # パラメータの変化率
        omega = reward \
                + self.gamma * np.max([self.q_table[next_state_index, j] for j in range(self.q_table.shape[1])]) \
                - self.q_table[state_index, action_index]

        # Qテーブルの更新
        self.q_table[state_index, action_index] = self.q_table[state_index, action_index] + self.lr * omega

    def eval(self, env,  n_episode=10,  seed=5, mean=True):
        rewards = []  # 各エピソードの累積報酬を格納する
        env.seed(seed)
        for e in range(n_episode):
            state = env.reset()
            reward_sum = 0. # 累積報酬
            while True:
                action = self.select_exploratory_action(state)
                # action = self.select_action(state)

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


# Actor-Critic


def test_tqagent():
    env = gym.make('Pendulum-v0')
    agent = TQLAgent(action_space=env.action_space, observation_space=env.observation_space,state_size=10, action_size=9,
                 lr=3e-4, gamma=0.99, eps=0.05)
    print(agent.q_table.shape)
    print(agent._state_index(env.observation_space.low))
    print(agent._state_index(env.observation_space.high))
    print(agent._action_index(env.action_space.low))
    print(agent._action_index(env.action_space.high))

if __name__ == '__main__':
    test_tqagent()