# This script is supposed to test the implementation of RL learners
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rl_agent import RL_Agent
from rl_env import PG_Learner
from rl_env import TRPO_Learner


print tf.__version__
tf.reset_default_graph()

env = gym.make('CartPole-v1')
env.reset()

# pg = PG_Learner(rl_agent=RL_Agent("cartpole"), 
#                 game_env=env,
#                 discount=0.95, 
#                 batch_size=100, 
#                 lr=0.01)

# for i in range(100):
#     pg.step()


trpo = TRPO_Learner(rl_agent=RL_Agent("cartpole"), 
                    game_env=env,
                    discount=0.99, 
                    batch_size=100, 
                    trpo_delta=0.02,
                    line_search_option="max")

for i in range(100):
    trpo.step()
    