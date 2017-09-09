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

class Cartpole_Agent(RL_Agent):
    # Overwriting supposedly abstract RL_Agent class
    # All what is left is to actually provide the specific model to choose action
    # It is still implied that
    # 1) __init__ method defines all its variables in model_name scope
    # 2) the class has self.session, self.prob_layer and self.log_prob_layer methods
    # The remaining functionality needed in PG and TRPO learners is still defined in abstract base
    def __init__(self, model_name):
        RL_Agent.__init__(self, model_name)
        with tf.variable_scope(model_name):
            self.session = tf.Session()
            
            self.input_layer = tf.placeholder(shape=[None, 4], dtype=tf.float32)
            self.dense1_layer = tf.layers.dense(self.input_layer, 
                                                units=4, use_bias=True, 
                                                activation=tf.nn.relu, name="dense1_weights"
                                               )
            
            self.dense2_layer = tf.layers.dense(self.dense1_layer, 
                                                units=2, use_bias=True, 
                                                activation=tf.nn.relu, name="dense2_weights"
                                               ) 
            
            self.prob_layer = tf.maximum(tf.minimum(tf.nn.softmax(self.dense2_layer), 0.9999), 0.0001)
            self.log_prob_layer = tf.log(self.prob_layer)
                        
            self.session.run(tf.global_variables_initializer())


# Setting RL environment and running simulations
# pg = PG_Learner(rl_agent=Cartpole_Agent("cartpole"), 
#                 game_env=env,
#                 discount=0.95, 
#                 batch_size=100, 
#                 frame_cap=2000,
#                 lr=0.01)

# for i in range(100):
#     pg.step()


trpo = TRPO_Learner(rl_agent=Cartpole_Agent("cartpole"), 
                    game_env=env,
                    discount=0.99, 
                    batch_size=100, 
                    frame_cap=2000,
                    trpo_delta=0.01,
                    line_search_option="max")

for i in range(100):
    trpo.step()
    