import os
import random
import datetime
import time
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

from tqdm import tqdm
import matplotlib.pyplot as plt

from agent import ActorNet, CriticNet
from utils import Logger
from buffer import ReplayBuffer, collate_buffer

def get_args():
    parser = argparse.ArgumentParser(description='Actor-Criticの設定')

    parser.add_argument('--max-step', type=float, default=5e+5, help='(int) 最大ステップ数. default: 5e+5')
    parser.add_argument('--gamma', type=float, default=0.99, help='(float) 減衰率. default: 0.99')
    parser.add_argument('--lr', type=float, default=3e-4, help='(float) 学習率. default: 3e-4')
    parser.add_argument('--seeds', type=str, default='2', help='(str) 評価環境のseed値. e.g. --seed 2,3,4 . default: 2')
    parser.add_argument('--save-step', type=int, default=500, help='(int) モデルの保存タイミング')
    parser.add_argument('--expl', type=int, default=10000, help='(int) ランダム行動ステップ数')
    parser.add_argument('--device', type=int, default=-1, help='(int) デバイス. -1はcpu, 0以上はGPUの番号. default: -1')

    parser.add_argument('--eval-seed', type=int, default=5, help='(int) 学習環境のseed値. default: 5')
    parser.add_argument('--eval-step', type=int, default=100, help='(int) 評価のタイミング. default: 100')
    parser.add_argument('--er', action='store_true', help='経験再生.')
    parser.add_argument('--batch-size', type=int, default=256, help='(int) 経験再生におけるバッチサイズ. default: 256')

    args = parser.parse_args()

    args.seeds = list(map(int, args.seeds.split(',')))

    if args.device >= 0 and torch.cuda.is_available():
        args.device = f'cuda:{args.device}'
    else:
        args.device = 'cpu'


    return args



def main():
    args = get_args()
    # -- ハイパーパラメータ --
    param = dict()
    param['model'] = 'Actor-Critic'
    param['max_step'] = args.max_step
    param['gamma'] = args.gamma
    param['lr'] = args.lr
    param['batch_size'] = args.batch_size
    param['seeds'] = args.seeds
    param['eval_seed'] = args.eval_seed
    param['er'] = args.er
    param['expl'] = args.expl


    # -- 環境のインスタンス生成 --
    env = gym.make('Pendulum-v0')

    # -- 実験の日時とディレクトリ作成 --
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('./log/actor-critic', now)
    os.makedirs(save_dir, exist_ok=True)

    # -- ログの作成 --
    logger = Logger(os.path.join(save_dir, 'experiment.log'))
    logger.log(f'date : {now}')
    logger.log(f'save directory : {save_dir}')

    logger.log(' -- hyperparameter -- ')
    logger.log('{')
    for k, v in param.items():
        logger.log(f'    {k}: {v}')
    logger.log('}')

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    for seed in args.seeds:
        # seedの設定
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        # model
        actor = ActorNet(action_space=env.action_space, inplaces=state_dim, places=action_dim, hidden_dim=256)
        critic = CriticNet(inplaces=state_dim + action_dim, places=1, hidden_dim=256)
        if args.device != 'cpu':
            actor.to(args.device)
            critic.to(args.device)

        # optim
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)

        # replay buffer
        replay_buffer = ReplayBuffer(args.max_step)

        # -- start training
        state = env.reset()
        episode = 1
        save_episodes = []
        eval_rewards = []

        for t in tqdm(range(int(args.max_step))):
            if args.device == 'cpu':
                action = actor(torch.tensor(state, dtype=torch.float32).to(args.device))[0].detach().numpy()
            else:
                action = actor(torch.tensor(state, dtype=torch.float32).to(args.device))[0].cpu().detach().numpy()

            # action = actions[0].detach().numpy()
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            if done:
                state = env.reset()
                episode += 1
                # save model
                if episode % args.save_step == 0:
                    save_episodes.append(episode)
                    actor.save_models(os.path.join(save_dir, f'actor{episode}.pth'))
                    critic.save_models(os.path.join(save_dir, f'critic{episode}.pth'))

                # evaluation
                if episode % args.eval_step == 0:
                    eval_reward = evaluate(actor, env=gym.make('Pendulum-v0'),n_episode=10, seed=args.eval_seed, gamma=args.gamma, device=args.device)
                    eval_rewards.append(eval_reward)
                    actor.train()

            if len(replay_buffer) > args.batch_size and t > args.expl:
                train(actor, critic, actor_optim, critic_optim, replay_buffer, args)



def train(actor, critic, actor_optim, critic_optim, replay_buffer, args):
    actor.train()
    critic.train()

    states, actions, next_states, rewards, dones = collate_buffer(replay_buffer, args.batch_size)
    if args.device != 'cpu':
        states = states.to(args.device)
        actions = actions.to(args.device)
        next_states = next_states.to(args.device)
        rewards = rewards.to(args.device)
        # dones = dones.to(args.devuce)
    # critic trainings
    x = torch.cat([next_states, actor(states)], dim=1).to(args.device)
    delta = rewards + args.gamma * critic(x)
    x = torch.cat([states, actions], dim=1).to(args.device)
    critic_loss = torch.pow(delta - critic(x), 2).mean()
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # actor training
    x = torch.cat([states, actor(states)], dim=1)
    actor_loss = critic(x).mean()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()


def evaluate(actor, env, n_episode, seed, gamma, device):
    rewards = []  # 各エピソードの累積報酬を格納する
    env.seed(seed)
    actor.eval()
    with torch.no_grad():
        for e in range(n_episode):
            state = env.reset()
            reward_sum = 0.  # 累積報酬
            while True:
                if device == 'cpu':
                    action = actor(torch.tensor(state, dtype=torch.float32).to(device))[0].detach().numpy()
                else:
                    action = actor(torch.tensor(state, dtype=torch.float32).to(device))[0].cpu().detach().numpy()

                next_state, reward, done, info = env.step(action)
                reward_sum += gamma * reward
                state = next_state
                if done:
                    break
            rewards.append(reward_sum)
    env.close()
    rewards = np.array(rewards)
    return rewards



if __name__ == '__main__':
    main()