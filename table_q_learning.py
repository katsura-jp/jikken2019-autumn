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

def get_args():
    parser = argparse.ArgumentParser(description='テーブルQ学習の設定')

    parser.add_argument('--max-step', type=int, default=5e+5, help='最大ステップ数')
    parser.add_argument('--state-size', type=int, default=10, help='Qテーブルの状態空間の分割数')
    parser.add_argument('--action-size', type=int, default=9, help='Qテーブルの行動空間の分割数')
    parser.add_argument('--gamma', type=float, default=0.99, help='減衰率')
    parser.add_argument('--lr', type=float, default=3e-4, help='学習率')
    parser.add_argument('--eps', type=float, default=0.05, help='eps-greedyの確率')
    parser.add_argument('--seeds', type=str, default='2', help='評価環境のseed値')


    parser.add_argument('--eval-seed', type=float, default=5, help='学習環境のseed値')
    parser.add_argument('--experiment-reply', action='store_true', help='経験再生')
    parser.add_argument('--batch-size', type=int, default=256, help='経験再生におけるバッチサイズ')


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
    exp_reply = args.experiment_reply


    # -- 環境のインスタンス生成 --
    env = gym.make('Pendulum-v0')

    # -- 実験の日時とディレクトリ作成 --
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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

    # obs_high = np.array([1., 1., 1.])
    # observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

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

        if not exp_reply:
            save_episodes = []

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
            # 経験再生(未実装)
            raise NotImplementedError
            replay_buffer = ReplayBuffer(args.max_step)
            replay_period = int(max_step) // 3
            episode = 1
            state = env.reset()
            for t in tqdm(range(int(max_step))):
                if episode % 500 == 0:
                    save_episodes.append(episode)
                    agent.save_models(os.path.join(save_dir, f'qtable_{seed}_{episode}.pickle'))

                action = agent.select_exploratory_action(state)
                next_state, reward, done, info = env.step(action)

                replay_buffer.add(state, action, next_state, reward, done) # バッファーに追加
                
                agent.train(state, action, next_state, reward, done)
                state = next_state

                if done:
                    state = env.reset()
                    episode += 1


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