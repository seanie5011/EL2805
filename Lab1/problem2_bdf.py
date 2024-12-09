import os
import random
import pickle
from typing import Tuple, List

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

def fourier_indices(order:int, state_dim:int)->np.ndarray:
    return np.indices([order + 1] * state_dim).reshape(state_dim, -1).T

def fourier_basis(state:np.ndarray, indices:np.ndarray, bounds:tuple[np.ndarray, np.ndarray])->np.ndarray:
    # Normalize the state to [0,1]^n
    s = (state - bounds[0]) / (bounds[1] - bounds[0])
    return np.cos(np.pi * indices @ s)

def epsilon_greedy(features:np.ndarray, weights:np.ndarray, epsilon:float)->int:
    if random.random() < epsilon:
        return np.random.randint(weights.shape[0])
    return np.argmax(weights @ features)

def sarsa_lambda(env:gym.Env, episodes:int, alpha:float, gamma:float, lmbda:float, epsilon:float, order:int, max_steps:int,
                 momentum:float=0.0, clip_val:float=5.0, nesterov:bool=False, decay_rate:float=1.0)->Tuple[List[float], np.ndarray, np.ndarray]:
    bounds = (env.observation_space.low, env.observation_space.high)
    state_dim = len(bounds[0])
    indices = fourier_indices(order, state_dim)
    n_features = (order + 1) ** state_dim

    # Compute scaled alphas for each feature: alpha_i = alpha / norm(η_i)
    norms = np.linalg.norm(indices, axis=1)
    scaled_alphas = np.where(norms == 0, alpha, alpha / norms)

    # Initialize weights
    # W is of shape (A, m) where A = number of actions, m = number of features
    weights = np.random.normal(0, 1/np.sqrt(n_features), (env.action_space.n, n_features))
    # weights = np.zeros((env.action_space.n, n_features))  # seems to be better
    # Momentum terms
    v = np.zeros_like(weights)

    rewards = []
    progress_bar = tqdm(range(episodes))
    for episode in progress_bar:
        state, _ = env.reset()
        features = fourier_basis(state, indices, bounds)
        action = epsilon_greedy(features, weights, epsilon)

        # Eligibility traces for each action-feature pair
        z = np.zeros_like(weights)
        
        total_reward = 0
        for _ in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            if done or truncated:
                break
            
            next_features = fourier_basis(next_state, indices, bounds)
            next_action = epsilon_greedy(next_features, weights, epsilon)

            # TD error
            td_error = reward + gamma * (weights[next_action] @ next_features) - (weights[action] @ features)

            # Update eligibility traces:
            # First decay all traces
            z *= gamma * lmbda
            # Increase trace for the visited (action, features)
            z[action] += features

            # Optionally clip the eligibility traces
            if clip_val is not None:
                np.clip(z, -clip_val, clip_val, out=z)

            # Update weights with momentum/Nesterov
            grad = td_error * z * scaled_alphas  # shape (A, m)
            
            v = momentum * v + grad
            weights += momentum * v + grad
            
            # Decay learning rate
            scaled_alphas *= decay_rate

            features, action = next_features, next_action
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        progress_bar.set_description(f"Training Ep: {episode+1}, Reward: {total_reward:.2f}")

    return rewards, weights, indices

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    order = 2
    episodes, max_steps = 500, 10000
    alpha, gamma, lmbda, epsilon = 0.1, 1.0, 0.9, 0.2
    momentum = 0.0  # Could try 0.9 for more stability
    nesterov = True     # Enable Nesterov acceleration
    decay_rate = 0.999  # Decay rate for learning rate
    
    rewards, weights, indices = sarsa_lambda(
        env, episodes, alpha, gamma, lmbda, epsilon, order, max_steps, 
        momentum=momentum, nesterov=nesterov, decay_rate=decay_rate
    )
    
    # Plot training curve
    window = 100
    plt.figure()
    # plt.plot(rewards, label="Episodic Rewards")
    running_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.plot(np.arange(len(running_avg)) + window-1, running_avg, label=f"Running Average (window={window})", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.grid()
    os.makedirs("figs/problem2", exist_ok=True)
    plt.savefig("figs/problem2/problem2_bdf.png")

    # Save weights for check_solution.py (Part f)
    # W is weights, N is indices (η)
    W = weights
    N = indices
    data = {'W': W, 'N': N}
    with open("weights.pkl", "wb") as f:
        pickle.dump(data, f)

    # For parts (d), (e):
    # -------------------------------------------------
    # (d1) The training curve is already plotted above.
    #
    # (d2) 3D plot of value function:
    # To plot the value function of the optimal policy, we need to fix a policy (greedy w.r.t. Q)
    # and evaluate V(s) = max_a Q(s,a). We can do a grid over the state space and evaluate:
    positions = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 50)
    velocities = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 50)
    V_grid = np.zeros((50, 50))
    Pi_grid = np.zeros((50, 50))

    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            ftrs = fourier_basis(np.array([pos, vel]), indices, (env.observation_space.low, env.observation_space.high))
            Q_values = weights @ ftrs
            best_a = np.argmax(Q_values)
            V_grid[i, j] = np.max(Q_values)
            Pi_grid[i, j] = best_a

    # 3D plot of value function
    X, Y = np.meshgrid(velocities, positions)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V_grid, cmap='viridis')
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Position")
    ax.set_zlabel("Value")
    ax.set_title("Value Function of the Learned Policy")
    plt.savefig("figs/problem2/value_function.png")

    # (d3) 3D plot of the policy:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Pi_grid, cmap='viridis')
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Position")
    ax.set_zlabel("Best Action")
    ax.set_title("Optimal Policy")
    plt.savefig("figs/problem2/policy.png")

    # (d4) Including η = [0,0]:
    # This corresponds to the constant basis function. If included, it often helps represent a base value.
    # Rerun training excluding [0,0] or including it to see differences.

    # (e) Experiments with α, λ:
    # You would run multiple training loops with different α, λ values, record final performance and plot.
    #
    # Different Q-initializations can be tested by changing the weights initialization.
    #
    # Custom exploration strategies:
    # For instance, a decreasing ε over time or a Boltzmann exploration using Q-values.

    # The code above and comments guide you through solving the rest of the tasks.

    os.system("python Lab1/check_solution.py")