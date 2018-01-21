# This script is supposed to test the implementation of RL learners
import gym
import numpy as np
import tensorflow as tf
import itertools
import sys
sys.path.append("..")

from rl_agent import RL_Agent
from rl_learner import PG_Learner
from rl_learner import TRPO_Learner


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

n_reruns = 100

lrs = [0.01, 0.1]
batch_sizes = [10, 25, 100]
frame_caps = [None, 2000]
discounts = [0.95, 0.99]
properties = [lrs, batch_sizes, frame_caps, discounts]
properties_names = ["lr", "batch_size", "frame_cap", "discount"]
for x in itertools.product(*properties):

    # Determining file to store simulations results
    out_filename = "simulations/cartpole_pg" 
    for i in range(len(x)):
        out_filename += "_" + properties_names[i] + "_" + str(x[i])
    out_filename += ".txt"

    # Determining the number of gradient steps
    grad_steps = 10000 / x[1]

    for i in range(n_reruns):
        #Setting RL environment and running simulations
        print out_filename
        print "Rerun #", i + 1
        tf.reset_default_graph()
        pg = PG_Learner(rl_agent=Cartpole_Agent("cartpole"), 
                        game_env=env,
                        discount=x[3], 
                        batch_size=x[1], 
                        frame_cap=x[2],
                        lr=x[0])

        for i in range(grad_steps):
            pg.step()

        with open(out_filename, "a") as f:
            f.write(" ".join(str(x) for x in pg.reward_history) + "\n")

# out_filename = "simulations/cartpole_trpo_delta_0.01_batchsize_100_framecap_2000_discount_0.99.txt"
# n_reruns = 100

# for i in range(n_reruns):
    # tf.reset_default_graph()
    # trpo = TRPO_Learner(rl_agent=Cartpole_Agent("cartpole"), 
    #                     game_env=env,
    #                     discount=0.99, 
    #                     batch_size=100, 
    #                     frame_cap=2000,
    #                     trpo_delta=0.01,
    #                     line_search_option="max")

    # for i in range(100):
    #     trpo.step()

    # with open(out_filename, "a") as f:
    #     f.write(" ".join(str(x) for x in trpo.reward_history) + "\n")
