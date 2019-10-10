import os
import random

import numpy as np
import torch
import gym
from tqdm import tqdm

from agent import TQLAgent as Agent
from utils import seed_setting

def main():
    seed = 2
    env = gym.make('Pendulum-v0')

    env.seed(seed)
    seed_setting(seed)

    agent = Agent(action_space=env.action_space, observation_space=env.observation_space)
    # state = env.reset()
    # for t in tqdm(range(1000)):
    #     action = agent.select_exploratory_action(state)
    #     next_state, reward, done, info = env.step(action)
    #     agent.train(state, action, next_state, reward, done)
    #     state = next_state
    #     if done:
    #         state = env.reset()
    env.close()