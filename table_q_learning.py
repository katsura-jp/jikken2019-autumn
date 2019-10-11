import os
import random
import datetime
import time

import numpy as np
import torch
import gym
from tqdm import tqdm

from agent import TQLAgent as Agent
from utils import seed_setting

# TODO: ハイパーパラメータの設定をargparserまたはjsonで置き換え
# TODO: ハイパーパラメータをconfig.jsonとかに保存
# TODO: logファイルの追加

def main():
    # -- ハイパーパラメータ --
    max_step = 5e+5 # 最大ステップ数
    state_size = 10 # 状態空間の分割数
    action_size = 9 # 行動空間の分割数
    gamma = 0.99 # 減衰率
    lr = 3e-4 # 学習率
    batch_size = 256 # バッチサイズ
    eps = 0.05 # ϵ-greedyの確率
    seed = 2 # 学習環境のseed値
    eval_seed = 5 # 評価環境のseed値

    # -- 環境のインスタンス生成 --
    env = gym.make('Pendulum-v0')


    # -- 乱数シードの設定 --
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    # -- 実験の日時とディレクトリ作成 --
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('./log/q-learing', now)
    os.makedirs(save_dir, exist_ok=True)


    # -- エージェントのインスタンス生成 --
    agent = Agent(action_space=env.action_space, observation_space=env.observation_space,
                  state_size=state_size, action_size=action_size,
                  lr=lr, gamma=gamma, eps=eps)


    # -- 学習 --
    episode = 1
    state = env.reset()
    for t in range(int(max_step)):
        # if episode % 500 == 0:
        #     env.render()
        #     time.sleep(0.05)
        action = agent.select_exploratory_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train(state, action, next_state, reward, done)
        state = next_state

        if done:
            print(f'finish episode {episode} | step = {t}')
            state = env.reset()
            episode += 1

    env.close()

if __name__ == '__main__':
    main()