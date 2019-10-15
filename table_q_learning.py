import os
import random
import datetime
import time
import argparse
import logging

import numpy as np
import torch
import gym

from gym import spaces

from tqdm import tqdm
import matplotlib.pyplot as plt

from agent import TQLAgent as Agent
from buffer import ReplayBuffer

# TODO: ハイパーパラメータをconfig.jsonとかに保存
# TODO: 経験再生の追加(どこでenqueue/dequeueするの？)
# TODO: パラメータ保存
# TODO: 評価
# TODO: 学習時間の計測

def get_args():
    parser = argparse.ArgumentParser(description='テーブルQ学習の設定')

    parser.add_argument('--max-step', type=int, default=5e+5, help='(int) 最大ステップ数. default: 5e+5')
    parser.add_argument('--state-size', type=int, default=10, help='(int) Qテーブルの状態空間の分割数. default: 10')
    parser.add_argument('--action-size', type=int, default=9, help='(int) Qテーブルの行動空間の分割数. default: 9')
    parser.add_argument('--gamma', type=float, default=0.99, help='(float) 減衰率. default: 0.99')
    parser.add_argument('--lr', type=float, default=3e-4, help='(float) 学習率. default: 3e-4')
    parser.add_argument('--eps', type=float, default=0.05, help='eps-greedyの確率. default: 0.05')
    parser.add_argument('--seeds', type=str, default='2', help='(str) 評価環境のseed値. e.g. --seed 2,3,4 . default: 2')


    parser.add_argument('--eval-seed', type=int, default=5, help='(int) 学習環境のseed値. default: 5')
    parser.add_argument('--er', action='store_true', help='経験再生.')
    parser.add_argument('--batch-size', type=int, default=256, help='(int) 経験再生におけるバッチサイズ. default: 256')


    args = parser.parse_args()

    args.seeds = list(map(int, args.seeds.split(',')))
    return args


def main():
    args = get_args()
    # -- ハイパーパラメータ --
    max_step = args.max_step # 最大ステップ数
    state_size = args.state_size # 状態空間の分割数
    action_size = args.action_size # 行動空間の分割数
    gamma = args.gamma # 減衰率
    lr = args.lr # 学習率
    batch_size = args.batch_size # バッチサイズ
    eps = args.eps # ϵ-greedyの確率
    seeds = args.seeds # 学習環境のseed値
    eval_seed = args.eval_seed # 評価環境のseed値
    er = args.er


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
    logger = logging.getLogger("Log")
    logger.setLevel(logging.DEBUG)
    handler_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(save_dir, 'experiment.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)

    logger.info(f'date : {now}')
    logger.info(f'save directory : {save_dir}')


    for seed in seeds:
        # -- 乱数シードの設定 --
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # -- エージェントのインスタンス生成 --
        agent = Agent(action_space=env.action_space, observation_space=env.observation_space,
                      state_size=state_size, action_size=action_size,
                      lr=lr, gamma=gamma, eps=eps)


        # -- 学習 --

        # これdictで持った方いいかも{episode: 10, seed: 2, path: path/to/file}
        save_episodes = []
        if not args.er:
            episode = 1
            state = env.reset()
            for t in tqdm(range(int(max_step))):
                action = agent.select_exploratory_action(state)
                next_state, reward, done, info = env.step(action)
                agent.train(state, action, next_state, reward, done)
                state = next_state
                if done:
                    state = env.reset()
                    if episode % 500 == 0:
                        save_episodes.append(episode)
                        agent.save_models(os.path.join(save_dir, f'qtable_{seed}_{episode}.pickle'))
                    episode += 1



        else:
            replay_buffer = ReplayBuffer(args.max_step)
            episode = 1
            state = env.reset()
            for t in tqdm(range(int(max_step))):
                action = agent.select_exploratory_action(state)
                next_state, reward, done, info = env.step(action)

                replay_buffer.add(state, action, next_state, reward, done) # バッファーに追加
                
                agent.train(state, action, next_state, reward, done)
                state = next_state

                if done:
                    state = env.reset()
                    episode += 1
                    if episode % 500 == 0:
                        save_episodes.append(episode)
                        agent.save_models(os.path.join(save_dir, f'qtable_{seed}_{episode}.pickle'))

                if len(replay_buffer) > args.batch_size:
                    for _args in replay_buffer.sample(args.batch_size):
                        agent.train(*_args)

    env.close()
    # -- 評価 --
    rewards = []
    for seed in seeds:
        for episode in save_episodes:
            agent.load_models(os.path.join(save_dir, f'qtable_{seed}_{episode}.pickle'))
            reward = agent.eval(env=gym.make('Pendulum-v0'), n_episode=10, seed=5, mean=False)
            logger.info(f'episode {episode} : reward mean = {reward.mean()}')
            rewards.append(reward)

        # visualize
        plt.boxplot(rewards,labels=save_episodes,showfliers=False)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.savefig(os.path.join(save_dir, f'boxplot_{seed}.png'))
        plt.clf()

if __name__ == '__main__':
    main()
