import roboschool
import gym
import time


# env = gym.make('RoboschoolHumanoid-v1')
env = gym.make('Pendulum-v0')
state = env.reset()
for t in range(1000):
    env.render()
    time.sleep(0.1)
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        break
env.close()
