import tensorflow as tf
import numpy as np
from rl_utils import *

# Defining a class for RL agent: should be possible to use it with either
# policy gradient and trpo methods
class RL_Agent:
    # Currently RL_Agent is supposed to be an abstract class:
    # It provides implemetation of methods to calculates gradients for TRPO and PG methods
    # However, the actual NN config to choose actions is left for inherited classes
    def __init__(self, model_name):
        # Just specifying which methods are need to be defined / overwritten ib inherited classes
        self.model_name = model_name
        self.n_actions = None

        self.session = None
        self.input_layer = None
        self.prob_layer = None
        self.log_prob_layer = None

        self.state_value = None

    ###
    # Maintenance section
    ###

    def model_variables(self):
        return [x for x in tf.trainable_variables() if self.model_name in x.name]
    
    def model_size(self):
        var_sizes = [tf.size(x) for x in self.model_variables()]
        return self.session.run(tf.reduce_sum(var_sizes))
    
    def variable_size(self):
        var_sizes = [tf.size(x) for x in self.model_variables()]
        return self.session.run(var_sizes)

    ###
    # Actor section
    ###
            
    def predict(self, states):
        return self.session.run(self.prob_layer, feed_dict={self.input_layer: states})
    
    def pg_grad(self, states, actions, rewards, baselines=None):
        # Calculating base policy gradient
        # Return a sum of log_prob gradients weighted by discounted sum of future rewards
        action_mask = tf.one_hot(actions, depth=self.n_actions, on_value=1.0, off_value=0.0, axis=-1)
        picked_log_prob_actions = tf.reduce_sum(action_mask * self.log_prob_layer, axis=1)
        if baselines is None:
            weighted_log_prob_actions = picked_log_prob_actions * rewards
        else:
            weighted_log_prob_actions = picked_log_prob_actions * (rewards - baselines)
        grad_log_prob_actions = get_flattened_gradients(weighted_log_prob_actions, self.model_variables())
        return self.session.run(grad_log_prob_actions, feed_dict={self.input_layer: states})

    ###
    # Critic section
    ###
            
    def evaluate_states(self, states):
        return self.session.run(self.state_value, feed_dict={self.input_layer: states})

    def critic_grad(self, states, rewards):
        # Calculating gradient of a critic agent
        # The implied loss is (R - V(s)) ** 2
        critic_loss = tf.reduce_sum(tf.square(rewards - self.state_value))
        grad_loss = get_flattened_gradients(critic_loss, self.model_variables())
        return self.session.run(grad_loss, feed_dict={self.input_layer: states})

    ###
    # TRPO section
    ###

    def trpo_grad(self, states, actions, rewards):
        # Calculating the target gradient as defined in section 5 of the trpo paper
        # Takes a gradient of prob action ratio weighted by Q-values
        # Essentially it should be a gradient for compare_prob_ratios function
        action_mask = tf.one_hot(actions, depth=self.n_actions, on_value=1.0, off_value=0.0, axis=-1)
        fixed_prob_actions = tf.stop_gradient(self.prob_layer)
        prob_ratio = self.prob_layer / fixed_prob_actions
        masked_prob_ratio = tf.reduce_sum(action_mask * prob_ratio, axis=1)
        weighted_prob_ratio = masked_prob_ratio * rewards
        trpo_gradient = get_flattened_gradients(weighted_prob_ratio, self.model_variables())
        return self.session.run(trpo_gradient, feed_dict={self.input_layer: states})
    
    def fisher_vector_product(self, states, vector):
        # This function is supposed to return the product of estimated fisher information matrix and a specified vector
        # As I hope to reliably estimate this matrix, I take all states accumulated in a batch of games
        expected_log_prob = tf.reduce_sum(tf.stop_gradient(self.prob_layer) * self.log_prob_layer, axis=1)
        log_prob_grad = get_flattened_gradients(expected_log_prob, self.model_variables())
        grad_vector_product = tf.reduce_sum(log_prob_grad * vector)
        fisher_vector_product = -get_flattened_gradients(grad_vector_product, self.model_variables()) / states.shape[0]
        return self.session.run(fisher_vector_product, feed_dict={self.input_layer: states})        
    
    def compare_prob_ratios(self, states, actions, rewards, d_var):
        # Estimates the function to optimize as defined is section 5 of original paper
        # I take the ratio of probabilities from original weights and probs from proposed weights (var + d_var)
        # These ratios are weighted by Q-values: thus, if output > 1 new weights make better actions more probable
        # Also, wighout scaling by original probs, its gradient is given by grad_log_prob_actions
        action_mask = tf.one_hot(actions, depth=self.n_actions, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)
        
        original_prob_actions = tf.reduce_sum(action_mask * self.prob_layer, axis=1)
        np_original_prob_actions = self.session.run(original_prob_actions, feed_dict={self.input_layer: states})

        for (grad, var) in zip(d_var, self.model_variables()):
            self.session.run(tf.assign_add(var, grad))
            
        new_prob_actions = tf.reduce_sum(action_mask * self.prob_layer, axis=1)   
        np_new_prob_actions = self.session.run(new_prob_actions, feed_dict={self.input_layer: states})

        for (grad, var) in zip(d_var, self.model_variables()):
            self.session.run(tf.assign_sub(var, grad))
            
        return sum((np_new_prob_actions / np_original_prob_actions) * np.array(rewards))

    def estimate_kl_divergence(self, states, d_var):
        # This function calculates an actual kl_divergence between action prob distributions
        # with current weights and weight + d_var
        original_probs = self.session.run(self.prob_layer, feed_dict={self.input_layer: states})
        original_log_probs = self.session.run(self.log_prob_layer, feed_dict={self.input_layer: states})

        for (grad, var) in zip(d_var, self.model_variables()):
            self.session.run(tf.assign_add(var, grad))
            
        new_log_probs = self.session.run(self.log_prob_layer, feed_dict={self.input_layer: states})

        for (grad, var) in zip(d_var, self.model_variables()):
            self.session.run(tf.assign_sub(var, grad))
            
        kl = original_probs * (original_log_probs - new_log_probs)
        return np.sum(kl) / kl.shape[0]
                