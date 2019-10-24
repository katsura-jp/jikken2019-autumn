import numpy as np
import pickle
import gym
import torch
import torch.nn as nn
import torch.distributions.normal as normal


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

        self._init_q_table()
        # Qテーブル Q[idx(s), idx(a)]
        print(f'state space dimention : {self.observation_space.shape[0]}')
        print(f'action space dimention : {self.action_space.shape[0]}')
        print(f'Q table size : {self.q_table.shape}')
        # indexの参照
        # action_index_refはactionの部分空間でもある

        action_margin = (self.action_space.high - self.action_space.low) / (self.action_size*2 + 1)
        state_margin = (self.observation_space.high - self.observation_space.low) / (self.state_size*2 + 1)
        self._action_index_ref = np.linspace(self.action_space.low + action_margin, self.action_space.high - action_margin, self.action_size)
        self._state_index_ref  = np.linspace(self.observation_space.low + state_margin, self.observation_space.high - state_margin, self.state_size)
        self._action_power = np.power(self.action_size, np.arange(self.action_space.shape[0]))
        self._state_power = np.power(self.state_size, np.arange(self.observation_space.shape[0]))

        #NOTE: デバッグ用
        # Qテーブルの更新回数を保存
        self.update_q_table = np.zeros(self.q_table.shape, dtype=np.uint)
        self.update_state = np.zeros(self.q_table.shape[0], dtype=np.uint)
        self.update_action = np.zeros(self.q_table.shape[1], dtype=np.uint)

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
        index = np.argmax(self.q_table[state_index,:])
        return self._action_index_ref[index]

    def select_exploratory_action(self, state):
        '''行動方策からアクションをサンプルする。'''
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
        delta = reward \
                + self.gamma * np.max(self.q_table[next_state_index,:]) \
                - self.q_table[state_index, action_index]

        # Qテーブルの更新
        self.q_table[state_index, action_index] = self.q_table[state_index, action_index] + self.lr * delta

        # 更新回数をインクリメント
        self.update_q_table[state_index, action_index] += 1
        self.update_state[state_index] += 1
        self.update_action[action_index] += 1


    def eval(self, env, n_episode=10, seed=5, mean=True):
        rewards = []  # 各エピソードの累積報酬を格納する
        env.seed(seed)
        for e in range(n_episode):
            state = env.reset()
            reward_sum = 0. # 累積報酬
            while True:
                # action = self.select_exploratory_action(state)
                action = self.select_action(state)

                next_state, reward, done, info = env.step(action)
                reward_sum += self.gamma * reward
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

class ActorNet(nn.Module):
    '''actionを返すモデル'''
    def __init__(self, action_space, inplaces, places, hidden_dim=256, omega=0.1):
        '''
        :param action_space: 行動部分空間
        :param inplaces: 入力の次元数(stateの次元)
        :param places: 出力の次元(actionの次元)
        :param hidden_dim: 隠れ層の次元数
        '''
        super(ActorNet, self).__init__()

        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.omega = omega

        # ノイズ
        # d = torch.diag(torch.tensor(self.action_space.high - self.action_space.low) * omega / 2).type(torch.float32)
        d = (torch.tensor(self.action_space.high - self.action_space.low) * omega / 2).type(torch.float32)
        self.norm = normal.Normal(torch.zeros(d.shape), d)

        # NN(Actorは2層)
        self.fc1 = nn.Linear(inplaces, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, places)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = ActorCriticTanh(self.action_space.high, self.action_space.low)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        if self.training:
            x = x + self._get_noise(x.shape[0]).to(x.device)

        x = self.clamp(x)
        return x

    def _get_noise(self, bs):
        noise = self.norm.sample(sample_shape=torch.Size([bs]))
        return noise

    def clamp(self, x):
        for i in range(x.shape[1]):
            x[:,i].clamp_(self.action_space.low[i], self.action_space.high[i])
        return x

    def save_models(self, path):
        torch.save(self.state_dict(), path)

    def load_models(self, path):
        self.load_state_dict(torch.load(path))


class CriticNet(nn.Module):
    '''Q値を返すモデル'''
    def __init__(self, inplaces, places, hidden_dim=256):
        '''
        :param inplaces: 入力の次元数(stateの次元 + actionの次元)
        :param places: 出力の次元(1次元)
        :param hidden_dim: 隠れ層の次元数
        '''
        super(CriticNet, self).__init__()

        # NN(Actorは2層)
        self.fc1 = nn.Linear(inplaces, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, places)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def save_models(self, path):
        torch.save(self.state_dict(), path)

    def load_models(self, path):
        self.load_state_dict(torch.load(path))


class ActorCriticTanh(nn.Module):
    def __init__(self, high, low):
        super(ActorCriticTanh, self).__init__()
        self.high = torch.tensor(high).type(torch.float32)
        self.low = torch.tensor(low).type(torch.float32)
        self.tanh = nn.Tanh()

    def forward(self, x):
        high = self.high.to(x.device)
        low = self.low.to(x.device)
        x = (high + low) / 2 + ((high - low) / 2) * self.tanh(x)
        return x


class ActorCriticAgent(Agent):
    def __init__(self, action_space, observation_space, optim, lr, gamma, device):
        self.action_space = action_space
        self.observation_space = observation_space

        action_dim = action_space.shape[0]
        state_dim = observation_space.shape[0]

        self.actor = ActorNet(action_space=action_space, inplaces=state_dim, places=action_dim, hidden_dim=256).to(device)
        self.critic = CriticNet(inplaces=state_dim + action_dim, places=1, hidden_dim=256).to(device)

        if optim == 'adam':
            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        elif optim == 'sgd':
            self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=lr)
            self.critic_optim = torch.optim.SGD(self.critic.parameters(), lr=lr)
        elif optim == 'momentum_sgd':
            self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
            self.critic_optim = torch.optim.SGD(self.critic.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
        else:
            raise NotImplementedError

        self.device = device
        self.lr = lr
        self.gamma = gamma


        self.critic_loss = None
        self.actor_loss = None


    def save_models(self, path):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()}, path)

    def load_models(self, path):
        weights = torch.load(path)
        self.actor.load_state_dict(weights['actor'])
        self.critic.load_state_dict(weights['critic'])

    def select_action(self, state):
        self.actor.eval()
        return self.actor(state)

    def select_exploratory_action(self, state):
        self.actor.train()
        return self.actor(state)

    def train(self, state, action, next_state, reward, done):
        self.actor.train()
        self.critic.train()

        if self.device != 'cpu':
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            reward = reward.to(self.device)

        # critic trainings
        x = torch.cat([next_state, self.actor(state)], dim=1).to(self.device)
        delta = reward + self.gamma * self.critic(x)
        x = torch.cat([state, action], dim=1).to(self.device)
        critic_loss = torch.pow(delta - self.critic(x), 2).mean()  # MSE
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor training
        x = torch.cat([state, self.actor(state)], dim=1)
        actor_loss = - self.critic(x).mean()  # SGDとプラマイ逆
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


        self.critic_loss = critic_loss.item()
        self.actor_loss = actor_loss.item()

    def eval(self, env, n_episode, seed):
        rewards = []  # 各エピソードの累積報酬を格納する
        env.seed(seed)
        self.actor.eval()
        with torch.no_grad():
            for e in range(n_episode):
                state = env.reset()
                reward_sum = 0.  # 累積報酬
                while True:
                    if self.device == 'cpu':
                        action = self.actor(torch.tensor([state], dtype=torch.float32).to(self.device))[0].detach().numpy()
                    else:
                        action = self.actor(torch.tensor([state], dtype=torch.float32).to(self.device))[0].cpu().detach().numpy()

                    next_state, reward, done, info = env.step(action)
                    reward_sum += self.gamma * reward
                    state = next_state
                    if done:
                        break
                rewards.append(reward_sum)
        env.close()
        rewards = np.array(rewards)
        return rewards

# --- test codes ---

def test_agent():
    import datetime
    import os

    env = gym.make('Pendulum-v0')
    agent = ActorCriticAgent(action_space=env.action_space, observation_space=env.observation_space,
                  optim='adam', lr=3e-4, gamma=0.99, device='cpu')
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('./log/test_agent', now)
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f'actor_critic.pth')
    agent.save_models(path)
    print('save model')
    agent.load_models(path)
    print('load model')





if __name__ == '__main__':
    test_agent()
    pass