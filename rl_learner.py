import tensorflow as tf
import numpy as np
from rl_utils import *
 

# Defining classes for RL environemt:
# They are supposed to take an instance of rl_agent, an instance of gym environment
# and provide a wrapper on top of rl_agent training

# An abstract class supposed to store rl_env information and actually play the games
class RL_Learner:
    
    def __init__(self, session, rl_agent, game_env, discount=0.99, batch_size=50, frame_cap=None):
        # Session is tensorflow session, "borrowed" from rl_agent definition
        # rl_agent is the NN choosing actions + wrapper for grad calculations
        # env is gym environment: black box for (state, action) -> (next_state, reward)
        self.session = session
        self.agent = rl_agent
        self.env = game_env
        
        # discount - discount for normalization of future rewards
        # batch_size - number of separate games in a batch (same NN weights)
        # frame_cap - the cap on the number of frames for which reward, action, state are returned
        # frame_cap may be needed for PG and (especially) TRPO to speed up computations
        self.discount = discount
        self.batch_size = batch_size
        self.frame_cap = frame_cap
        
        self.reward_history = []
        self.played_games = 0
        
    def play_single_game(self):
        states = []
        actions = []
        rewards = []
        
        observation = np.expand_dims(self.env.reset(), axis=0)
        done = False
        
        while done == False:
            states.append(observation)
            prob_actions = self.agent.predict(observation)[0]
            action = np.random.choice(np.arange(len(prob_actions)), p=prob_actions)
            actions.append(action)
            observation, reward, done, info = self.env.step(action)
            observation = np.expand_dims(observation, axis=0)
            rewards.append(reward)
            
        states = np.concatenate(states, axis=0)
        actions = np.array(actions).reshape((-1, 1))
        rewards = np.array(rewards).reshape((-1, 1))
        self.reward_history.append(np.sum(rewards))
        self.played_games += 1
            
        return states, actions, rewards
    
    def play_batch(self):        
        all_states = []
        all_actions = []
        all_rewards = []
        
        for i in range(self.batch_size):

            states, actions, rewards = self.play_single_game()
            
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(sum_discounted_rewards(rewards, self.discount))
            
        print "Average reward for batch #", self.played_games / self.batch_size, \
              ": ", sum(self.reward_history[-self.batch_size:]) / float(self.batch_size)
        
        concat_states = np.concatenate(all_states, axis=0)
        concat_actions = np.concatenate(all_actions, axis=0)
        concat_rewards = np.concatenate(all_rewards, axis=0)

        # Selecting a subset of all frames uniformly at random if there is a cap
        if self.frame_cap is not None:
            picked_frames = np.random.choice(concat_states.shape[0], size=self.frame_cap, replace=True)
            concat_states = concat_states[picked_frames, :]
            concat_actions = concat_actions[picked_frames, :]
            concat_rewards = concat_rewards[picked_frames, :]
        
        return concat_states, concat_actions, concat_rewards

# Implementation of policy gradient learner
class PG_Learner(RL_Learner):
    
    def __init__(self, session, rl_agent, game_env, discount=0.99, batch_size=50, frame_cap=None, lr=0.1):
        RL_Learner.__init__(self, session, rl_agent, game_env, discount, batch_size, frame_cap)
        self.lr = lr
        
    def step(self):
        # One policy gradient step based on one batch of games
        
        concat_states, concat_actions, concat_rewards = self.play_batch()
        grad_reward = self.agent.pg_grad(concat_states,
                                         concat_actions,
                                         concat_rewards)  / self.batch_size
        
        grads = unflatten_gradient(tf.constant(self.lr * grad_reward, dtype=tf.float32), self.agent.model_variables())
        for (grad, var) in zip(grads, self.agent.model_variables()):
            self.session.run(tf.assign_add(var, grad))  

# Implementation of A3C learner
class A3C_Learner(RL_Learner):
    
    def __init__(self, session, actor_agent, critic_agent, game_env, 
                 discount=0.99, batch_size=50, frame_cap=None, actor_lr=0.01, critic_lr=0.01):
        RL_Learner.__init__(self, session, actor_agent, game_env, discount, batch_size, frame_cap)
        self.critic_agent = critic_agent
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def step(self):
        # One policy gradient step based on one batch of games
        
        concat_states, concat_actions, concat_rewards = self.play_batch()
        concat_baselines = self.critic_agent.evaluate_states(concat_states)

        # Gradient step for actor
        grad_reward = self.agent.pg_grad(concat_states,
                                         concat_actions,
                                         concat_rewards,
                                         concat_baselines)  / self.batch_size
        
        grads = unflatten_gradient(tf.constant(self.actor_lr * grad_reward, dtype=tf.float32), self.agent.model_variables())
        for (grad, var) in zip(grads, self.agent.model_variables()):
            self.session.run(tf.assign_add(var, grad)) 

        # Gradient step for critic
        critic_grad = self.critic_agent.critic_grad(concat_states,
                                                    concat_rewards)  / self.batch_size
        
        grads = unflatten_gradient(tf.constant(self.critic_lr * critic_grad, dtype=tf.float32), self.critic_agent.model_variables())
        for (grad, var) in zip(grads, self.critic_agent.model_variables()):
            self.session.run(tf.assign_sub(var, grad)) 

# Implementation of TRPO learner
class TRPO_Learner(RL_Learner):
    
    def __init__(self, session, rl_agent, game_env, discount, batch_size, frame_cap=None, 
                 trpo_delta=0.01, line_search_option="max"):
        RL_Learner.__init__(self, session, rl_agent, game_env, discount, batch_size, frame_cap)
        self.trpo_delta = trpo_delta
        # Line search options could be: 
        # * "none": trpo step is taken without checking actual KL-divergence
        # * "max": trpo step is maximal in picked direction that satisfies KL-constraint
        # * "best": trpo step is the one that satisfies the constraint and gives the best prob ratio along the line
        self.line_search_option = line_search_option
        
    def scale_down_grads(self, obs_states, grads):
        # This function adjusts trpo direction until it starts to satisfy KL-constraint
        while self.agent.estimate_kl_divergence(obs_states, grads) > self.trpo_delta:
            grads = [0.5 * grad for grad in grads]
        return grads
    
    def line_search(self, obs_states, obs_actions, obs_rewards, grads):
        # This function shrinks grads in various ways and picks the scaling that:
        # * satisfies KL-constraint
        # * gives the best weighted action prob ratio
        deltas = [2 ** (-i) for i in range(10)] + [0]

        def feasibility_check(delta):
            trunc_grads = [delta * grad for grad in grads]
            return self.agent.estimate_kl_divergence(obs_states, trunc_grads) <= self.trpo_delta
        deltas = filter(feasibility_check, deltas)
            
        def prob_ratio(delta):
            trunc_grads = [delta * grad for grad in grads]
            return self.agent.compare_prob_ratios(obs_states, obs_actions, obs_rewards, trunc_grads)
        
        best_delta = max(deltas, key=prob_ratio)
        return [best_delta * grad for grad in grads]
            
    def step(self):
        
        concat_states, concat_actions, concat_rewards = self.play_batch()
        
        grad_reward = self.agent.trpo_grad(concat_states,
                                           concat_actions,
                                           concat_rewards)  / self.batch_size
        
        Ax_fun = lambda x: self.agent.fisher_vector_product(concat_states, tf.constant(x, dtype=tf.float32))
        
        trpo_dir = cg_solver(Ax_fun, grad_reward)
        scaling = np.sqrt(2 * self.trpo_delta / np.sum(trpo_dir * Ax_fun(trpo_dir)))
        
        grads = unflatten_gradient(tf.constant(scaling * trpo_dir, dtype=tf.float32), self.agent.model_variables())
               
        if self.line_search_option == "max":
            grads = self.scale_down_grads(concat_states, grads)
        elif self.line_search_option == "best":
            grads = self.line_search(concat_states, concat_actions, concat_rewards, grads)  
                
        for (grad, var) in zip(grads, self.agent.model_variables()):
            self.session.run(tf.assign_add(var, grad))
