import gym
import tensorflow as tf
import numpy as np
import skimage.transform
import scipy.ndimage.filters
import matplotlib.pyplot as plt
import imageio

env = gym.make('Pong-v0')

def parse_observation(ob):
    # The function that takes the original frame from Pong game and transforms it:
    # grayscale, cropping, blurring and resizing are performed
    ob_gray = np.dot(ob[...,:3], [0.299, 0.587, 0.114])
    ob_crop = ob_gray[34:194, :]
    ob_blur = scipy.ndimage.filters.gaussian_filter(ob_crop, 5)
    ob_resize = skimage.transform.resize(ob_blur, (32, 32), order=0)
    return np.expand_dims(ob_resize, axis=-1)

class Custom_Pong_Env:
    # The main of this custom environment is to preprocess the images one receives from original env
    # Currently is simply a placeholder
    def __init__(self, proper_gym_env):
        self.gym_env = proper_gym_env
        
    def reset(self):
        ob = self.gym_env.reset()
        return parse_observation(ob)
    
    def step(self, action):
        ob, reward, done, info = self.gym_env.step(action) 
        return parse_observation(ob), reward, done, info  

import sys
sys.path.append("..")
from rl_agent import RL_Agent
from rl_learner import TRPO_Learner

class Pong_Agent(RL_Agent):
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

            self.input_layer = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)
            self.conv_1 = tf.layers.conv2d(self.input_layer, filters=8, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
            self.pool_1 = tf.layers.max_pooling2d(self.conv_1, pool_size=3, strides=2, padding="same")

            self.conv_2 = tf.layers.conv2d(self.pool_1, filters=16, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
            self.pool_2 = tf.layers.max_pooling2d(self.conv_2, pool_size=3, strides=2, padding="same")

            self.conv_3 = tf.layers.conv2d(self.pool_2, filters=32, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
            self.pool_3 = tf.layers.max_pooling2d(self.conv_3, pool_size=3, strides=2, padding="same")

            self.flat = tf.contrib.layers.flatten(self.pool_3)
            self.dense_1 = tf.layers.dense(self.flat, units=10, activation=tf.nn.relu)
            self.dense_2 = tf.layers.dense(self.dense_1, units=2)
                        
            self.prob_layer = tf.maximum(tf.minimum(tf.nn.softmax(self.dense_2), 0.9999), 0.0001)
            self.log_prob_layer = tf.log(self.prob_layer)
                        
            self.session.run(tf.global_variables_initializer())

tf.reset_default_graph()
trpo = TRPO_Learner(rl_agent=Pong_Agent("2018_01_23_pong_model"), 
                    game_env=Custom_Pong_Env(env),
                    discount=0.99, 
                    batch_size=25,
                    frame_cap=25,
                    trpo_delta=0.02,
                    line_search_option="max")

for i in range(1):
    trpo.step()