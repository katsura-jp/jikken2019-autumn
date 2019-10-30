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
    def __init__(self, action_space, inplaces, places, hidden_dim=256, n_layer=2):
        super(ActorNet, self).__init__()

        self.action_space = action_space
        self.hidden_dim = hidden_dim

        # NN(Actorは2層)
        layers = []
        layers.append(nn.Linear(inplaces, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for n in range(1, n_layer-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, places))
        layers.append(ActorCriticTanh(self.action_space.high, self.action_space.low))
        self.layers = nn.Sequential(*layers)

        self.fc1 = nn.Linear(inplaces, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, places)
        self.relu = nn.ReLU(inplace=True)
        self.activation = ActorCriticTanh(self.action_space.high, self.action_space.low)


    def forward(self, x):
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.activation(x)
        x = self.layers(x)
        return x



class CriticNet(nn.Module):
    def __init__(self, inplaces, places, hidden_dim=256, n_layer=3):
        super(CriticNet, self).__init__()

        # NN(Actorは2層)
        layers = []
        layers.append(nn.Linear(inplaces, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for n in range(1, n_layer-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, places))
        self.layers = nn.Sequential(*layers)

        self.fc1 = nn.Linear(inplaces, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, places)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.layers(x)
        return x


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
    def __init__(self, action_space, observation_space, optim, lr, gamma, device='cpu', sigma=0.1,
                 critic_layer=3, actor_layer=2, critic_hidden_dim=256, actor_hidden_dim=256):
        self.action_space = action_space
        self.observation_space = observation_space

        action_dim = action_space.shape[0]
        state_dim = observation_space.shape[0]

        self.actor = ActorNet(action_space=action_space, inplaces=state_dim, places=action_dim, hidden_dim=actor_hidden_dim, n_layer=actor_layer).to(device)
        self.actor.apply(self._init_agent)
        self.critic = CriticNet(inplaces=state_dim + action_dim, places=1, hidden_dim=critic_hidden_dim, n_layer=critic_layer).to(device)
        self.critic.apply(self._init_agent)

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

        # ノイズの生成
        d = (torch.tensor(self.action_space.high - self.action_space.low) * sigma / 2).type(torch.float32)
        self.norm = normal.Normal(torch.zeros(d.shape), d)

    def _clamp(self, x):
        for i in range(x.shape[1]):
            x[:,i].clamp_(self.action_space.low[i], self.action_space.high[i])
        return x

    def _get_noise(self, bs):
        noise = self.norm.sample(sample_shape=torch.Size([bs]))
        return noise

    def _init_agent(self, m):
        # Xavier initialization
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def save_models(self, path):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()}, path)

    def load_models(self, path):
        weights = torch.load(path)
        self.actor.load_state_dict(weights['actor'])
        self.critic.load_state_dict(weights['critic'])

    def select_action(self, state):
        x = self.actor(state)
        return x

    def select_exploratory_action(self, state):
        x = self.select_action(state)
        x = x + self._get_noise(x.shape[0]).to(x.device)
        x = self._clamp(x)
        return x

    def train(self, state, action, next_state, reward, done):
        self.actor.train()
        self.critic.train()

        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)

        # critic trainings
        x = torch.cat([next_state, self.select_action(state)], dim=1).to(self.device)
        delta = reward + self.gamma * self.critic(x)
        x = torch.cat([state, action], dim=1).to(self.device)
        critic_loss = torch.pow(delta - self.critic(x), 2).mean()  # MSE
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor training
        x = torch.cat([state, self.select_action(state)], dim=1)
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
                        action = self.select_action(torch.tensor([state], dtype=torch.float32).to(self.device))[0].detach().numpy()
                    else:
                        action = self.select_action(torch.tensor([state], dtype=torch.float32).to(self.device))[0].cpu().detach().numpy()

                    next_state, reward, done, info = env.step(action)
                    reward_sum += self.gamma * reward
                    state = next_state
                    if done:
                        break
                rewards.append(reward_sum)
        env.close()
        rewards = np.array(rewards)
        return rewards


class TD3Agent(Agent):
    def __init__(self, action_space, observation_space, optim, lr, gamma, sigma_beta=0.1,
                 target_ac=True, smooth_reg=True, delay_update=True, clip_double=True,
                 tau=0.005, clip=0.5, delay=2, sigma_target=0.2,
                 critic_layer=3, actor_layer=2, critic_hidden_dim=256, actor_hidden_dim=256,
                 device='cpu'):

        self.action_space = action_space
        self.observation_space = observation_space

        action_dim = action_space.shape[0]
        state_dim = observation_space.shape[0]

        self.actor = ActorNet(action_space=action_space, inplaces=state_dim, places=action_dim, hidden_dim=actor_hidden_dim, n_layer=actor_layer).to(device)
        self.actor.apply(self._init_agent)
        self.critic = CriticNet(inplaces=state_dim + action_dim, places=1, hidden_dim=critic_hidden_dim, n_layer=critic_layer).to(device)
        self.critic.apply(self._init_agent)

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
        self.sigma_beta = sigma_beta # 行動方策のノイズ
        self.lr = lr # 学習率
        self.gamma = gamma # 減衰率
        self.tau = tau # Target Actor&Critic の更新割合
        self.clip = clip # smooth regularizationのノイズのclip
        self.delay = delay # delayed policy updateの頻度
        self.sigma_target = sigma_target # smooth regularizationのノイズ


        self.critic_loss = None
        self.actor_loss = None

        # ノイズの生成
        d = (torch.tensor(self.action_space.high - self.action_space.low) * sigma_beta / 2).type(torch.float32)
        self.norm = normal.Normal(torch.zeros(d.shape), d)

        # -- TD3 の 工夫 --
        self.target_ac = target_ac # Target Actor & Target Critic
        self.smooth_reg = smooth_reg # Target Policy Smoothing Regularization
        self.delay_update = delay_update # Delayed Policy Update
        self.clip_double = clip_double # Clipped Double Q-Learning

        # 条件によるオブジェクトの生成
        if self.target_ac: # Target Actor & Target Critic
            self.target_actor = ActorNet(action_space=action_space, inplaces=state_dim, places=action_dim, hidden_dim=actor_hidden_dim, n_layer=actor_layer).to(device)
            self.target_critic = CriticNet(inplaces=state_dim + action_dim, places=1, hidden_dim=critic_hidden_dim, n_layer=critic_layer).to(device)
            self._init_target_agent(self.actor, self.target_actor)
            self._init_target_agent(self.critic, self.target_critic)

        if self.smooth_reg:
            self.target_actor_noise = normal.Normal(torch.zeros(d.shape), torch.zeros(d.shape).fill_(sigma_target))
            pass

        if not self.delay_update:
            self.delay = 1
        self.delay_count = 0

        if self.clip_double:
            self.critic2 = CriticNet(inplaces=state_dim + action_dim, places=1, hidden_dim=critic_hidden_dim, n_layer=critic_layer).to(device)
            self.critic2.apply(self._init_agent)
            if self.target_ac:
                self.target_critic2 = CriticNet(inplaces=state_dim + action_dim, places=1, hidden_dim=critic_hidden_dim, n_layer=critic_layer).to(device)
                self._init_target_agent(self.critic2, self.target_critic2)

            self.critic2_loss = None
            if optim == 'adam':
                self.critic2_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
            elif optim == 'sgd':
                self.critic2_optim = torch.optim.SGD(self.critic.parameters(), lr=lr)
            elif optim == 'momentum_sgd':
                self.critic2_optim = torch.optim.SGD(self.critic.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
            else:
                raise NotImplementedError


    def _clamp(self, x):
        for i in range(x.shape[1]):
            x[:, i].clamp_(self.action_space.low[i], self.action_space.high[i])
        return x

    def _get_noise(self, bs):
        noise = self.norm.sample(sample_shape=torch.Size([bs]))
        return noise

    def _init_agent(self, m):
        # Xavier initialization
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def _init_target_agent(self, agent, target):
        target.load_state_dict(agent.state_dict())

    def _update_target(self, agent, target):
        for agent_param, target_param in zip(agent.parameters(), target.parameters()):
            target_param.data = (1. - self.tau) * target_param.data + self.tau * agent_param.data

    def _get_target_actor_noise(self, bs):
        noise = self.target_actor_noise.sample(sample_shape=torch.Size([bs])).clamp(-self.clip, self.clip)
        return noise

    def save_models(self, path):
        weights = {'actor': self.actor.state_dict(),
                   'critic': self.critic.state_dict()}

        if self.target_ac:
            weights['target_actor'] = self.target_actor.state_dict()
            weights['target_critic'] = self.target_critic.state_dict()
        if self.clip_double:
            weights['critic2'] = self.critic2.state_dict()
            if self.target_ac:
                weights['target_critic2'] = self.target_critic2.state_dict()

        torch.save(weights, path)

    def load_models(self, path):
        weights = torch.load(path)
        self.actor.load_state_dict(weights['actor'])
        self.critic.load_state_dict(weights['critic'])

        if self.clip_double:
            self.critic2.load_state_dict(weights['critic2'])
            if self.target_ac:
                self.target_actor.load_state_dict(weights['target_actor'])
                self.target_critic.load_state_dict(weights['target_critic'])
                self.target_critic2.load_state_dict(weights['target_critic2'])
        elif self.target_ac:
            self.target_actor.load_state_dict(weights['target_actor'])
            self.target_critic.load_state_dict(weights['target_critic'])

    def select_action(self, state):
        x = self.actor(state)
        return x

    def select_exploratory_action(self, state):
        x = self.select_action(state)
        x = x + self._get_noise(x.shape[0]).to(x.device)
        x = self._clamp(x)
        return x

    def train(self, state, action, next_state, reward, done):
        self.actor.train()
        self.critic.train()

        if self.target_ac:
            self.target_actor.train()
            self.target_critic.train()
        if self.clip_double:
            self.critic2.train()
            if self.target_ac:
                self.target_critic2.train()

        self.delay_count += 1

        # deviceの設定
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)


        # critic trainings

        # next actionの計算
        if self.target_ac:
            next_action = self.target_actor(state)
            if self.smooth_reg:
                next_action = next_action + self._get_target_actor_noise(next_action.shape[0]).to(next_action.device)
                next_action = self._clamp(next_action)
        else:
            next_action = self.select_action(state)

        # ターゲットの計算
        x = torch.cat([next_state, next_action], dim=1).to(self.device)
        if self.clip_double:
            if self.target_ac:
                delta = reward + self.gamma * torch.stack([self.target_critic(x), self.target_critic2(x)], dim=0).min(dim=0).values
            else:
                delta = reward + self.gamma * torch.stack([self.critic(x), self.critic2(x)], dim=0).min(dim=0).values
        elif self.target_ac:
            delta = reward + self.gamma * self.target_critic(x)
        else:
            delta = reward + self.gamma * self.critic(x)

        x = torch.cat([state, action], dim=1).to(self.device)
        critic_loss = torch.pow(delta - self.critic(x), 2).mean()  # MSE

        if self.clip_double:
            # critic2の損失計算
            delta2 = delta.clone().detach()
            x2 = x.clone().detach()
            critic2_loss = torch.pow(delta2 - self.critic2(x2), 2).mean()  # MSE

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            self.critic2_optim.step()

            self.critic_loss = critic_loss.item()
            self.critic2_loss = critic2_loss.item()
        else:
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            self.critic_loss = critic_loss.item()


        if self.delay_count == self.delay:
            # actor training
            x = torch.cat([state, self.select_action(state)], dim=1)
            actor_loss = - self.critic(x).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.actor_loss = actor_loss.item()
            if self.target_ac:
                self._update_target(self.actor, self.target_actor)
                self._update_target(self.critic, self.target_critic)
                if self.clip_double:
                    self._update_target(self.critic2, self.target_critic2)
            self.delay_count = 0


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
                        action = self.select_action(torch.tensor([state], dtype=torch.float32).to(self.device))[
                            0].detach().numpy()
                    else:
                        action = self.select_action(torch.tensor([state], dtype=torch.float32).to(self.device))[
                            0].cpu().detach().numpy()

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