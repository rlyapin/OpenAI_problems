import tensorflow as tf
import numpy as np

# The utilities functions for all rl involved projects:

def var_size(v):
    return int(np.prod([int(d) for d in v.shape]))

def get_padded_gradients(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return [g if g is not None else tf.zeros(v.shape)
            for g, v in zip(grads, var_list)]

def get_flattened_gradients(loss, var_list):
    padded_gradients = get_padded_gradients(loss, var_list)
    return tf.concat([tf.reshape(x, [-1]) for x in padded_gradients], 0)

def unflatten_gradient(grad, var_list):
    shapes = [v.shape for v in var_list]
    sizes = [var_size(v) for v in var_list]
    grads = []

    pointer = 0
    for (shape, size, v) in zip(shapes, sizes, var_list):
        grads.append(tf.reshape(grad[pointer:pointer + size], shape))
        pointer += size
    return grads

def sum_discounted_rewards(rewards, discount):
    discounted_rewards = list(rewards)
    pointer = len(rewards) - 1
    acc_discounted_sum = rewards[-1]
    while pointer > 0:
        acc_discounted_sum *= discount
        pointer -= 1
        discounted_rewards[pointer] += acc_discounted_sum
        acc_discounted_sum += rewards[pointer]
    return discounted_rewards

def cg_solver(Ax_fun, b, k_iter=10, eps=10**(-6)):
    # Solver for Ax = b that uses conjugate gradient method
    # Ax_fun is supposed to be a function that takes x and returns Ax
    # Impementation of the solver is taken from Wikipedia article
    # Solver itself is supposed to be outside tf realm - all vector inside are np arrays
    # Note that it is assumed k_iter < dim(b)
    x = np.zeros(b.shape, dtype=np.float)
    r = b - Ax_fun(x)
    p = np.array(r)
    for k in range(k_iter):
        
        r_norm = float(np.sum(r * r))
        A_p = Ax_fun(p)
        alpha = r_norm / np.sum(p * A_p)
        x += alpha * p
        r -= alpha * A_p
        next_r_norm = np.sum(r * r)
        
        if next_r_norm < eps:
            break
            
        beta = next_r_norm / r_norm
        p *= beta
        p += r
        
    return x   