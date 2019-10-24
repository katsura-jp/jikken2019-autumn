import os
import random
import datetime
import time
import argparse
import logging
import pickle

import numpy as np
import torch
import gym

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from agent import TQLAgent as Agent
from buffer import ReplayBuffer
from utils import Logger

# TODO: ハイパーパラメータをconfig.jsonとかに保存
# TODO: 経験再生の追加(どこでenqueue/dequeueするの？)
# TODO: パラメータ保存
# TODO: 評価
# TODO: 学習時間の計測

def get_args():
    parser = argparse.ArgumentParser(description='テーブルQ学習の設定')

    parser.add_argument('--max-step', type=float, default=5e+5, help='(int) 最大ステップ数. default: 5e+5')
    parser.add_argument('--state-size', type=int, default=10, help='(int) Qテーブルの状態空間の分割数. default: 10')
    parser.add_argument('--action-size', type=int, default=9, help='(int) Qテーブルの行動空間の分割数. default: 9')
    parser.add_argument('--gamma', type=float, default=0.99, help='(float) 減衰率. default: 0.99')
    parser.add_argument('--lr', type=float, default=3e-4, help='(float) 学習率. default: 3e-4')
    parser.add_argument('--eps', type=float, default=0.05, help='eps-greedyの確率. default: 0.05')
    parser.add_argument('--seeds', type=str, default='2', help='(str) 評価環境のseed値. e.g. --seed 2,3,4 . default: 2')
    parser.add_argument('--save-step', type=int, default=500, help='(int) モデルの保存タイミング. default: 500')

    parser.add_argument('--eval-seed', type=int, default=5, help='(int) 学習環境のseed値. default: 5')
    parser.add_argument('--eval-step', type=int, default=100, help='(int) 評価のタイミング. default: 100')
    parser.add_argument('--er', action='store_true', help='経験再生.')
    parser.add_argument('--batch-size', type=int, default=256, help='(int) 経験再生におけるバッチサイズ. default: 256')

    args = parser.parse_args()

    args.seeds = list(map(int, args.seeds.split(',')))
    return args


def main():
    args = get_args()
    # -- ハイパーパラメータ --
    param = dict()
    param['model'] = 'TQLAgent'
    param['max_step'] = args.max_step
    param['state_size'] = args.state_size
    param['action_size'] = args.action_size
    param['gamma'] = args.gamma
    param['lr'] = args.lr
    param['batch_size'] = args.batch_size
    param['eps'] = args.eps
    param['seeds'] = args.seeds
    param['eval_seed'] = args.eval_seed
    param['er'] = args.er


    # -- 環境のインスタンス生成 --
    env = gym.make('Pendulum-v0')

    # -- 実験の日時とディレクトリ作成 --
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.er:
        save_dir = os.path.join('./log/q-learing-er', now)
    else:
        save_dir = os.path.join('./log/q-learing', now)
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

    for seed in args.seeds:
        # -- 乱数シードの設定 --
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # -- エージェントのインスタンス生成 --
        agent = Agent(action_space=env.action_space, observation_space=env.observation_space,
                      state_size=args.state_size, action_size=args.action_size,
                      lr=args.lr, gamma=args.gamma, eps=args.eps)

        # -- 学習 --
        eval_rewards = []
        save_episodes = []
        if not args.er:
            episode = 1
            state = env.reset()
            for t in tqdm(range(int(args.max_step))):
                # if episode > 2000:
                #     import pdb; pdb.set_trace()
                action = agent.select_exploratory_action(state)
                next_state, reward, done, info = env.step(action)
                agent.train(state, action, next_state, reward, done)
                state = next_state
                if done:
                    # reset environment
                    state = env.reset()
                    episode += 1

                    # save model
                    if episode % args.save_step == 0:
                        save_episodes.append(episode)
                        agent.save_models(os.path.join(save_dir, f'qtable_{seed}_{episode}.pickle'))

                    # evaluation
                    if episode % args.eval_step == 0:
                        eval_reward = agent.eval(env=gym.make('Pendulum-v0'), n_episode=10, seed=args.eval_seed, mean=False)
                        eval_rewards.append(eval_reward)
        else:
            replay_buffer = ReplayBuffer(args.max_step)
            episode = 1
            state = env.reset()
            for t in tqdm(range(int(args.max_step))):
                action = agent.select_exploratory_action(state)
                next_state, reward, done, info = env.step(action)

                replay_buffer.add(state, action, next_state, reward, done) # バッファーに追加
                state = next_state

                if done:
                    # reset environment
                    state = env.reset()
                    episode += 1

                    # save model
                    if episode % args.save_step == 0:
                        save_episodes.append(episode)
                        agent.save_models(os.path.join(save_dir, f'qtable_{seed}_{episode}.pickle'))

                    # evaluation
                    if episode % args.eval_step == 0:
                        eval_reward = agent.eval(env=gym.make('Pendulum-v0'), n_episode=10, seed=args.eval_seed, mean=False)
                        eval_rewards.append(eval_reward)

                # experience replay
                if len(replay_buffer) > args.batch_size:
                    for _args in replay_buffer.sample(args.batch_size):
                        agent.train(*_args)

        # visualize
        plot(eval_rewards, args.eval_step, os.path.join(save_dir, f'plot_{seed}.png'))
        with open(os.path.join(save_dir, f'history_{seed}.pickle'), 'wb') as f:
            pickle.dump(eval_rewards, f)

    env.close()
    # -- 評価 --
    for seed in args.seeds:
        rewards = []
        for episode in save_episodes:
            agent.load_models(os.path.join(save_dir, f'qtable_{seed}_{episode}.pickle'))
            reward = agent.eval(env=gym.make('Pendulum-v0'), n_episode=10, seed=5, mean=False)
            logger.log(f'episode {episode} : reward mean = {reward.mean()}')
            rewards.append(reward)
            heatmap(agent.q_table, os.path.join(save_dir, f'qtable_{seed}_{episode}.png'))

        # visualize
        boxplot(rewards, save_episodes, os.path.join(save_dir, f'boxplot_{seed}.png'))
        heatmap(agent.update_q_table, os.path.join(save_dir, f'update_qtable_{seed}.png'))
        hist(agent.update_state, os.path.join(save_dir, f'update_state_{seed}.png'))
        hist(agent.update_action, os.path.join(save_dir, f'update_action_{seed}.png'))


def boxplot(rewards, labels, path):
    plt.boxplot(rewards, labels=labels, showfliers=False)
    plt.title('reward history')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig(path)
    plt.clf()

def plot(rewards, step, path):
    rewards = np.array(rewards)
    x = (np.arange(rewards.shape[0]) + 1) * step
    y = np.percentile(rewards, [0, 25, 50, 75, 100], axis=1)
    plt.plot(x, y[0], color='black') # min
    plt.plot(x, y[4], color='black') # max
    plt.fill_between(x, y[1], y[3], color='gray', alpha=0.5) # 25%, 75%
    plt.plot(x, y[2], color='red') # mean
    plt.title('reward history')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig(path)
    plt.clf()


def heatmap(x, path):
    sns.heatmap(x)
    plt.savefig(path)
    plt.clf()

def hist(y, path):
    x = np.arange(len(y))
    plt.hist(x, weights=y, bins=len(x), rwidth=0.8)
    plt.savefig(path)
    plt.clf()


if __name__ == '__main__':
    main()
