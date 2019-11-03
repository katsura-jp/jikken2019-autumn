import os
import random
import datetime
import argparse
import json

import torch
import tensorboardX as tbx
import numpy as np
import roboschool
import gym
import pickle

import matplotlib.pyplot as plt

from agent import ActorCriticAgent as Agent
from utils import Logger
from buffer import ReplayBuffer, collate_buffer

def get_args():
    parser = argparse.ArgumentParser(description='Actor-Criticの設定')

    parser.add_argument('--max-step', type=float, default=5e+5, help='(int) 最大ステップ数. default: 5e+5')
    parser.add_argument('--gamma', type=float, default=0.99, help='(float) 減衰率. default: 0.99')
    parser.add_argument('--lr', type=float, default=3e-4, help='(float) 学習率. default: 3e-4')
    parser.add_argument('--seed', type=int, default=2, help='(int) 評価環境のseed値. default: 2')
    parser.add_argument('--save-step', type=int, default=500, help='(int) モデルの保存タイミング')
    parser.add_argument('--expl', type=int, default=10000, help='(int) ランダム行動ステップ数')
    parser.add_argument('--device', type=int, default=-1, help='(int) デバイス. -1はcpu, 0以上はGPUの番号. default: -1')

    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd', 'momentum_sgd'],
                        help='(str) 最適化関数. adam, sgd, momentum_sgdから選択. default: adam')
    parser.add_argument('--sigma-beta', type=float, default=0.1, help='(float) ⾏動⽅策のノイズ. default: 0.1')

    parser.add_argument('--actor-layer', type=int, default=2, help='(int) Actorのネットワーク層の数. default: 2')
    parser.add_argument('--critic-layer', type=int, default=3, help='(int) Criticのネットワーク層の数. default: 3')
    parser.add_argument('--actor-dim', type=int, default=256, help='(int) Actorの中間層の次元数. default: 256')
    parser.add_argument('--critic-dim', type=int, default=256, help='(int) Criticの中間層の次元数. default: 256')

    parser.add_argument('--eval-seed', type=int, default=5, help='(int) 学習環境のseed値. default: 5')
    parser.add_argument('--eval-step', type=int, default=10000, help='(int) 評価のタイミング. default: 10000')
    parser.add_argument('--eval-episodes', type=int, default=10, help='(int) 評価のエピソード数. default: 10')
    parser.add_argument('--batch-size', type=int, default=256, help='(int) 経験再生におけるバッチサイズ. default: 256')
    parser.add_argument('--div-step', action='store_true', help='評価時の累積報酬和をstep数で破る.')

    parser.add_argument('--env', type=str, default='Pendulum-v0', help='(str) Gymにおける環境名. default: Pendulum-v0')
    args = parser.parse_args()

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
    param['seed'] = args.seed
    param['eval_seed'] = args.eval_seed
    param['eval_episodes'] = args.eval_episodes
    param['expl'] = args.expl
    param['optim'] = args.optim
    param['sigma_beta'] = args.sigma_beta
    param['env'] = args.env
    param['div_step'] = args.div_step

    param['actor_layer'] = args.actor_layer
    param['critic_layer'] = args.critic_layer
    param['actor_hidden_dim'] = args.actor_dim
    param['critic_hidden_dim'] = args.critic_dim

    # -- 環境のインスタンス生成 --
    env = gym.make(args.env)

    # -- 実験の日時とディレクトリ作成 --
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('./log/actor-critic', now)
    os.makedirs(save_dir, exist_ok=True)

    # ハイパーパラメータの保存
    with open(os.path.join(save_dir, 'hyperparam.json'), 'w') as f:
        json.dump(param, f, indent=4)


    # -- ログの作成 --
    logger = Logger(os.path.join(save_dir, 'experiment.log'))
    logger.log(f'date : {now}')
    logger.log(f'save directory : {save_dir}')

    logger.log(' -- hyperparameter -- ')
    logger.log('{')
    for k, v in param.items():
        logger.log(f'    {k}: {v}')
    logger.log('}')

    writer = tbx.SummaryWriter(save_dir)

    # seedの設定
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    agent = Agent(action_space=env.action_space, observation_space=env.observation_space,
                  optim=args.optim, lr=args.lr, gamma=args.gamma, device=args.device, sigma=args.sigma_beta,
                  actor_layer=args.actor_layer, critic_layer=args.critic_layer,
                  actor_hidden_dim=args.actor_dim, critic_hidden_dim=args.critic_dim)

    # replay buffer
    replay_buffer = ReplayBuffer(args.max_step)

    # -- start training
    state = env.reset()
    episode = 1
    save_episodes = []
    eval_rewards = []
    train_actor_loss = []
    train_critic_loss = []
    eval_reward_ = -10000.00

    logger.log('   Step   | Episode | L(actor) | L(critic) | reward ')

    for t in range(int(args.max_step)):
        if args.device == 'cpu':
            action = agent.select_exploratory_action(torch.tensor([state], dtype=torch.float32).to(args.device))[0].detach().numpy()
        else:
            action = agent.select_exploratory_action(torch.tensor([state], dtype=torch.float32).to(args.device))[0].cpu().detach().numpy()

        next_state, reward, done, info = env.step(action)
        replay_buffer.add(state, action, next_state, reward, done)
        state = next_state
        if done:
            state = env.reset()
            episode += 1
            # save model
            if episode % args.save_step == 0:
                save_episodes.append(episode)
                agent.save_models(os.path.join(save_dir, f'actor_critic_{episode}.pth'))

        # evaluation
        if t % args.eval_step == 0:
            eval_reward = agent.eval(env=gym.make(args.env), n_episode=args.eval_episodes,seed=args.eval_seed, mean=args.div_step)
            eval_rewards.append(eval_reward)
            writer.add_scalar("reward", eval_reward.mean(), episode)
            if agent.actor_loss is not None:
                print('\r', end='')
                logger.log(
                    f'{t:8}  | {episode:7} | {agent.actor_loss:.7} |  {agent.critic_loss:.7} | {eval_reward.mean():.3f}')
            eval_reward_ = eval_reward.mean()

        if len(replay_buffer) > args.batch_size and t > args.expl:
            states, actions, next_states, rewards, dones = collate_buffer(replay_buffer, args.batch_size)
            agent.train(states, actions, next_states, rewards, dones)

            writer.add_scalar("loss/critic_loss", agent.critic_loss, t)
            writer.add_scalar("loss/actor_loss", agent.actor_loss, t)
            train_critic_loss.append(agent.critic_loss)
            train_actor_loss.append(agent.actor_loss)
            if t % env._max_episode_steps == 0:
                print('\r', end='')
                print(f'\r{t:8}  | {episode:7} | {agent.actor_loss:.7} |  {agent.critic_loss:.7} | {eval_reward_:.3f}', end='')

    env.close()
    
    plot(eval_rewards, args.eval_step, os.path.join(save_dir, f'plot_{args.seed}.png'))
    with open(os.path.join(save_dir, f'history_{args.seed}.pickle'), 'wb') as f:
        pickle.dump(eval_rewards, f)
    with open(os.path.join(save_dir, f'actor_loss_{args.seed}.pickle'), 'wb') as f:
        pickle.dump(train_actor_loss, f)
    with open(os.path.join(save_dir, f'critic_loss_{args.seed}.pickle'), 'wb') as f:
        pickle.dump(train_critic_loss, f)

    rewards = []
    for episode in save_episodes:
        agent.load_models(os.path.join(save_dir, f'actor_critic_{episode}.pth'))
        reward = agent.eval(env=gym.make(args.env), n_episode=args.eval_episodes, seed=args.eval_seed, mean=args.div_step)
        rewards.append(reward)

    # visualize
    boxplot(rewards, save_episodes, os.path.join(save_dir, f'boxplot_{args.seed}.png'))

    

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

if __name__ == '__main__':
    main()