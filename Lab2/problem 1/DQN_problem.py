# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.

# Load packages
from collections import deque, namedtuple
import glob
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, DQNAgent
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# seed = 42
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# env.seed(seed)
# env.action_space.seed(seed)

# Define Experience tuple
# Experience represents a transition in the environment, including the current state, action taken,
# received reward, next state, and whether the episode is done.
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer:
    """Replay buffer for storing experiences.
    
       The experience replay buffer stores past experiences so that the agent can learn from them later.
       By sampling randomly from these experiences, the agent avoids overfitting to the most recent 
       transitions and helps stabilize training.
       - The buffer size is limited, and older experiences are discarded to make room for new ones.
       - Experiences are stored as tuples of (state, action, reward, next_state, done).
       - A batch of experiences is sampled randomly during each training step for updating the Q-values."""

    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=maximum_length)  # Using deque ensures efficient removal of oldest elements

    def append(self, experience):
        """Add a new experience to the buffer"""
        self.buffer.append(experience)

    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

    def sample_batch(self, n):
        """Randomly sample a batch of experiences"""
        if n > len(self.buffer):
            raise IndexError('Sample size exceeds buffer size!')
        indices = np.random.choice(len(self.buffer), size=n, replace=False)  # Random sampling
        batch = [self.buffer[i] for i in indices]  # Create a batch from sampled indices
        return zip(*batch)  # Unzip batch into state, action, reward, next_state, and done

def select_action(state, epsilon):
    """Epsilon-greedy action selection
    # We balance exploration and exploitation using epsilon-greedy.
    # Exploration: Choose a random action.
    # Exploitation: Choose the action with the highest Q-value (the optimal action)."""
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore by selecting a random action
    else:
        state_tensor = torch.tensor([state], dtype=torch.float32)  # Convert state to tensor
        return network(state_tensor).argmax().item()  # Exploit by selecting the action with max Q-value

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# storing models and configurations
model_dir = 'models'
config_path = 'models\\config.txt'

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v3', render_mode = "human")
env.reset()

# Parameters
N_episodes = 200  # Number of episodes
discount_factor = 0.99  # Value of the discount factor
n_ep_running_average = 50  # Running average of 50 episodes
n_actions = env.action_space.n  # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

epsilon_max = 0.95
epsilon_min = 0.05
epsilon = epsilon_max
epsilon_episodes = 500
batch_size = 32
buffer_size = 20000
learning_rate = 0.001
clip_max_norm = 0.5

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Initialize experience replay buffer
buffer = ExperienceReplayBuffer(maximum_length=buffer_size)
# Initialize the Q-network (state -> Q-values for actions)
network = DQNAgent(n_actions=n_actions, input_size=env.observation_space.shape[0], output_size=env.action_space.n)
# Optimizer for training the Q-network
optimizer = optim.Adam(network.parameters(), lr=learning_rate)  # Adam optimizer for efficient training

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.0
    t = 0
    while not (done or truncated):
        # Take a random action
        action = select_action(state, epsilon)

        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        # Store the experience (state, action, reward, next state, done) in the buffer
        buffer.append(Experience(state, action, reward, next_state, done))
        state = next_state
        t += 1

        # Training step: update Q-values using a batch of experiences from the buffer
        if len(buffer) >= (buffer_size / batch_size):
            # Sample a batch of experiences from the buffer
            states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)

            # Convert the batch data into tensors
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)  # Unsqueeze for correct shape
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Compute Q-values for the current states
            q_values = network(states).gather(1, actions).squeeze()  # Q-values for taken actions

            # Compute the target Q-values for the next states
            with torch.no_grad():  # No need to compute gradients for target Q-values
                next_q_values = network(next_states).max(1)[0]  # Max Q-value for next state
                targets = rewards + discount_factor * next_q_values * (1 - dones)  # Target: Bellman equation

            # Compute the loss (MSE loss between predicted Q-values and target Q-values)
            loss = nn.functional.mse_loss(q_values, targets)
               
            # Backpropagation step: update network parameters
            optimizer.zero_grad()  # Zero gradients before backpropagation
            loss.backward()  # Compute gradients
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_max_norm)  # Clip gradients to avoid exploding gradients
            optimizer.step()  # Update parameters

    # Decay epsilon: reduce exploration over time
    epsilon = max(epsilon_min, epsilon_max * (epsilon_min / epsilon_max)**((i) / (0.9 * epsilon_episodes - 1)))

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

# Close environment
env.close()

# save model
print('saving model...')
# find the next free filepath
filepaths = glob.glob(f'{model_dir}\\*')
base_string = f'{model_dir}\\model_'
i = 0
filepath = base_string + f'{i:06}.pth'
while filepath in filepaths:
    filepath = base_string + f'{i:06}.pth'
    i += 1
filename = filepath.split('\\')[-1]
# save the model and config details
torch.save(network.state_dict(), filepath)
with open(config_path, "a") as file:
    file.writelines(elem + '\n' for elem in [
        '----------',
        filename,
        'Architecture: ',  # example: "{input_size, 64, output_size} with ReLU activation on input and hidden layer"
        f'N_episodes: {N_episodes}',
        f'discount_factor: {discount_factor}',
        f'n_ep_running_average: {n_ep_running_average}',
        f'batch_size: {batch_size}',
        f'buffer_size: {buffer_size}',
        f'learning_rate: {learning_rate}',
        f'epsilon_max: {epsilon_max}',
        f'epsilon_min: {epsilon_min}',
        f'epsilon_episodes: {epsilon_episodes}',
        f'clip_max_norm: {clip_max_norm}',
        'final average reward: '
    ])
print(f'{filepath} saved with details in {config_path}')
print('model saved!')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
