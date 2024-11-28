# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 100        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma


# Reward
episode_reward_list = []  # Used to save episodes reward


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x


# Training process
for i in range(N_episodes):
    # Reset enviroment data
    done = False
    truncated = False
    state = scale_state_variables(env.reset()[0])
    total_episode_reward = 0.

    while not (done or truncated):
        # Take a random action
        # env.action_space.n tells you the number of actions
        # available
        action = np.random.randint(0, k)
            
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise. Truncated is true if you reach 
        # the maximal number of time steps, False else.
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = scale_state_variables(next_state)

        # Update episode reward
        total_episode_reward += reward
            
        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()
    

# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()