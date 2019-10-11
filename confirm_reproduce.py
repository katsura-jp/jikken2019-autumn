import roboschool
import gym

import os
import datetime
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from agent import RandomAgent as Agent

def main():
    # 実験の日時とディレクトリ作成
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('./log/conf-reproduce', now)
    os.makedirs(save_dir, exist_ok=True)
    print(f'save directory `{save_dir}`')
    # 乱数シードの設定
    seed = 2
    eval_seed = 5
    env = gym.make('Pendulum-v0')

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ハイパーパラメータ
    n_episode = 5
    n_step = 1000

    # エージェントの作成
    agent = Agent(action_space=env.action_space, observation_space=env.observation_space)

    # train step
    total_step = 0
    for episode in range(n_episode):
        state = env.reset()
        for t in range(n_step):
            # env.render()
            total_step += 1

            action = agent.select_exploratory_action(state)
            next_state, reward, done, info = env.step(action)
            agent.train(state, action, next_state, reward, done)
            state = next_state
            if done:
                break

        agent.save_models(os.path.join(save_dir, f'episode_{episode}.pickle'))
    env.close()

    # eval step
    logger = []
    for episode in range(n_episode):
        agent.load_models(os.path.join(save_dir, f'episode_{episode}.pickle'))
        eval_rewards = agent.eval(env=gym.make('Pendulum-v0'), n_episode=10,seed=eval_seed, mean=False)
        logger.append(eval_rewards)

    # visualize
    plt.boxplot(logger,labels=[e for e in range(n_episode)],showfliers=False)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()

if __name__ == '__main__':
    main()