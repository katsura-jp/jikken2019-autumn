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
from buffer import ReplayBuffer
class CriticLoss(nn.Module):
    def __init__(self):
        super(CriticLoss, self).__init__()
        pass
    def forward(self):
        pass

class ActorLoss(nn.Module):
    def __init__(self):
        super(ActorLoss, self).__init__()
        pass
    def forward(self):
        pass


def get_args():
    parser = argparse.ArgumentParser(description='テーブルQ学習の設定')

    parser.add_argument('--max-step', type=float, default=5e+5, help='(int) 最大ステップ数. default: 5e+5')
    parser.add_argument('--state-size', type=int, default=10, help='(int) Qテーブルの状態空間の分割数. default: 10')
    parser.add_argument('--action-size', type=int, default=9, help='(int) Qテーブルの行動空間の分割数. default: 9')
    parser.add_argument('--gamma', type=float, default=0.99, help='(float) 減衰率. default: 0.99')
    parser.add_argument('--lr', type=float, default=3e-4, help='(float) 学習率. default: 3e-4')
    parser.add_argument('--eps', type=float, default=0.05, help='eps-greedyの確率. default: 0.05')
    parser.add_argument('--seeds', type=str, default='2', help='(str) 評価環境のseed値. e.g. --seed 2,3,4 . default: 2')
    parser.add_argument('--save-step', type=int, default=500, help='(int) モデルの保存タイミング')

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
        actor = ActorNet(action_space=env.action_space, inplaces=state_dim, places=action_dim, hidden_dim=256)
        critic = CriticNet(inplaces=state_dim + action_dim, places=1, hidden_dim=256)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)
        replay_buffer = ReplayBuffer(args.max_step)




    pass




if __name__ == '__main__':
    main()