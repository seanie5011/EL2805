import os
import random

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

def fourier_basis(state:np.ndarray, order:int, bounds:tuple[np.ndarray, np.ndarray])->np.ndarray:
    state = (state - bounds[0]) / (bounds[1] - bounds[0])
    indices = np.indices([order + 1] * len(state)).reshape(len(state), -1).T
    return np.cos(np.pi * indices @ state)

def epsilon_greedy(features:np.ndarray, weights:np.ndarray, epsilon:float)->int:
    return np.random.randint(weights.shape[0]) if random.random() < epsilon else np.argmax(weights @ features)

def sarsa_lambda(env:gym.Env, episodes:int, alpha:float, gamma:float, lmbda:float, epsilon:float, order:int, max_steps:int)->list[float]:
    bounds = (env.observation_space.low, env.observation_space.high)
    n_features = (order + 1) ** len(bounds[0])
    weights = np.random.normal(0, 1/np.sqrt(n_features), (env.action_space.n, n_features))
    rewards = []

    progress_bar = tqdm(range(episodes))
    for episode in progress_bar:
        state, _ = env.reset()
        features = fourier_basis(state, order, bounds)
        action = epsilon_greedy(features, weights, epsilon)
        z = np.zeros_like(weights)
        total_reward = 0

        for _ in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            next_features = fourier_basis(next_state, order, bounds)
            next_action = epsilon_greedy(next_features, weights, epsilon)
            td_error = reward + gamma * (weights[next_action] @ next_features) - (weights[action] @ features)
            z[action] = gamma * lmbda * z[action] + features
            weights += alpha * td_error * z
            features, action = next_features, next_action
            total_reward += reward

            if done: break

        rewards.append(total_reward)
        progress_bar.set_description(f"Training Episodes (Reward: {total_reward:.2f})")

    return rewards

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    order = 2
    episodes, max_steps = 500, 10000
    alpha, gamma, lmbda, epsilon = 0.1, 1.0, 0.9, 0.1 
    rewards = sarsa_lambda(env, episodes, alpha, gamma, lmbda, epsilon, order, max_steps)
    plt.plot(rewards, label = "Episodic Rewards")
    plt.plot(np.convolve(rewards, np.ones(10) / 10, mode = "valid"), label = "Running Average")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.grid()

    os.makedirs("plots", exist_ok = True)
    plt.savefig("plots/problem2_bdf.png")

    plt.show()


