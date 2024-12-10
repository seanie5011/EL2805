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
    # weights = np.random.normal(0, 1/np.sqrt(n_features), (env.action_space.n, n_features))
    weights = np.zeros((env.action_space.n, n_features))  # seems to be better
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
        # z = np.random.normal(0, 1/np.sqrt(n_features), (env.action_space.n, n_features))
        
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

def evaluate_policy(env, weights, indices, n_episodes=50):
    """Evaluate a policy for n_episodes and return mean reward and std."""
    rewards = []
    bounds = (env.observation_space.low, env.observation_space.high)
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            features = fourier_basis(state, indices, bounds)
            action = np.argmax(weights @ features)  # Greedy policy
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if truncated:
                break
                
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)

def analyze_parameter_sensitivity(env, param_name, param_values, fixed_params):
    """Analyze sensitivity to a parameter by training multiple policies."""
    means, stds = [], []
    
    for value in tqdm(param_values, desc=f"Testing {param_name}"):
        params = fixed_params.copy()
        params[param_name] = value
        
        rewards, weights, indices = sarsa_lambda(
            env, 
            episodes=params['episodes'],
            alpha=params['alpha'],
            gamma=params['gamma'],
            lmbda=params['lambda'],
            epsilon=params['epsilon'],
            order=params['order'],
            max_steps=params['max_steps'],
            momentum=params['momentum']
        )
        
        mean_reward, std_reward = evaluate_policy(env, weights, indices)
        means.append(mean_reward)
        stds.append(std_reward)
        
    return np.array(means), np.array(stds)

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    order = 2
    episodes, max_steps = 500, 10000
    alpha, gamma, lmbda, epsilon = 1e-2, 1.00, 0.95, 0.1 # best values found in a previous run
    momentum = 0.0
    nesterov = True
    decay_rate = 0.999
    
    # Original training
    rewards, weights, indices = sarsa_lambda(
        env, episodes, alpha, gamma, lmbda, epsilon, order, max_steps, 
        momentum=momentum, nesterov=nesterov, decay_rate=decay_rate
    )
    
    # # (d1) Plot training curve with confidence intervals
    # window = 50
    # plt.figure(figsize=(10, 6))
    # running_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    # plt.plot(np.arange(len(running_avg)) + window-1, running_avg, label=f"Running Average", color='red')
    
    # # Add confidence intervals
    # std_dev = np.std(rewards)
    # plt.fill_between(
    #     np.arange(len(running_avg)) + window-1,
    #     running_avg - std_dev,
    #     running_avg + std_dev,
    #     alpha=0.2,
    #     color='red'
    # )
    
    # plt.xlabel("Episodes")
    # plt.ylabel("Total Reward")
    # plt.title("Training Performance with Confidence Intervals")
    # plt.legend()
    # plt.grid()
    # os.makedirs("figs/problem2", exist_ok=True)
    # plt.savefig("figs/problem2/training_curve.png")
    
    # # weights
    # data = {'W': weights, 'N': indices}
    # with open("weights.pkl", "wb") as f:
    #     pickle.dump(data, f)
        
        
    # os.system("python Lab1/check_solution.py")
    
    # # (e1) More detailed analysis of α
    # # Test more values in the likely optimal range
    # alphas = np.concatenate([
    #     np.logspace(-4, -3, 5),  # Very small learning rates
    #     np.logspace(-3, -2, 10),  # Small learning rates
    #     np.logspace(-2, -1, 10),  # Medium learning rates
    #     np.logspace(-1, 0, 5)     # Large learning rates
    # ])
    
    # print("Starting detailed alpha analysis...")
    # fixed_params = {
    #     'episodes': 200,
    #     'alpha': alpha,
    #     'gamma': gamma,
    #     'lambda': lmbda,
    #     'epsilon': epsilon,
    #     'order': order,
    #     'max_steps': max_steps,
    #     'momentum': momentum
    # }
    
    # alpha_means, alpha_stds = analyze_parameter_sensitivity(
    #     env, 'alpha', alphas, fixed_params
    # )
    
    # # Plot detailed alpha analysis
    # plt.figure(figsize=(12, 6))
    # plt.errorbar(alphas, alpha_means, yerr=alpha_stds, fmt='o-', capsize=5)
    # plt.xscale('log')
    # plt.xlabel('Learning Rate (α)')
    # plt.ylabel('Average Total Reward')
    # plt.title('Detailed Learning Rate Sensitivity Analysis')
    # plt.grid(True, which="both", ls="-", alpha=0.2)
    # plt.fill_between(alphas, alpha_means - alpha_stds, alpha_means + alpha_stds, alpha=0.2)
    
    # # Add vertical lines for different ranges
    # plt.axvline(x=1e-3, color='r', linestyle='--', alpha=0.3, label='Range Boundaries')
    # plt.axvline(x=1e-2, color='r', linestyle='--', alpha=0.3)
    # plt.axvline(x=1e-1, color='r', linestyle='--', alpha=0.3)
    
    # # Annotate the best alpha
    # best_alpha_idx = np.argmax(alpha_means)
    # best_alpha = alphas[best_alpha_idx]
    # plt.plot(best_alpha, alpha_means[best_alpha_idx], 'r*', markersize=15, label=f'Best α={best_alpha:.2e}')
    # plt.legend()
    # plt.savefig("figs/problem2/detailed_alpha_sensitivity.png")
    # plt.close()
    
    # print(f"Best alpha found: {best_alpha:.2e}")
    
    # # (e2) Detailed lambda analysis using the best alpha
    # print("\nStarting lambda analysis with best alpha...")
    # fixed_params['alpha'] = best_alpha  # Use the best alpha found
    
    # # Test more lambda values
    # lambdas = np.linspace(0, 1, 15)  # Test 15 evenly spaced values
    # lambda_means, lambda_stds = analyze_parameter_sensitivity(
    #     env, 'lambda', lambdas, fixed_params
    # )
    
    # plt.figure(figsize=(12, 6))
    # plt.errorbar(lambdas, lambda_means, yerr=lambda_stds, fmt='o-', capsize=5)
    # plt.xlabel('Lambda (λ)')
    # plt.ylabel('Average Total Reward')
    # plt.title(f'Lambda Sensitivity Analysis (with α={best_alpha:.2e})')
    # plt.grid(True)
    # plt.fill_between(lambdas, lambda_means - lambda_stds, lambda_means + lambda_stds, alpha=0.2)
    
    # # Annotate the best lambda
    # best_lambda_idx = np.argmax(lambda_means)
    # best_lambda = lambdas[best_lambda_idx]
    # plt.plot(best_lambda, lambda_means[best_lambda_idx], 'r*', markersize=15, label=f'Best λ={best_lambda:.2f}')
    # plt.legend()
    # plt.savefig("figs/problem2/detailed_lambda_sensitivity.png")
    # plt.close()
    
    # print(f"Best lambda found: {best_lambda:.2f}")
    
    # # After finding best alpha and lambda, analyze gamma
    # print("\nStarting gamma analysis with best alpha and lambda...")
    # fixed_params['lambda'] = best_lambda  # Use best lambda found
    
    # # Test gamma values
    # gammas = np.linspace(0.8, 1.0, 10)  # Focus on high gamma values as this is episodic
    # gamma_means, gamma_stds = analyze_parameter_sensitivity(
    #     env, 'gamma', gammas, fixed_params
    # )
    
    # plt.figure(figsize=(12, 6))
    # plt.errorbar(gammas, gamma_means, yerr=gamma_stds, fmt='o-', capsize=5)
    # plt.xlabel('Discount Factor (γ)')
    # plt.ylabel('Average Total Reward')
    # plt.title(f'Gamma Sensitivity Analysis (with α={best_alpha:.2e}, λ={best_lambda:.2f})')
    # plt.grid(True)
    # plt.fill_between(gammas, gamma_means - gamma_stds, gamma_means + gamma_stds, alpha=0.2)
    
    # # Annotate the best gamma
    # best_gamma_idx = np.argmax(gamma_means)
    # best_gamma = gammas[best_gamma_idx]
    # plt.plot(best_gamma, gamma_means[best_gamma_idx], 'r*', markersize=15, label=f'Best γ={best_gamma:.2f}')
    # plt.legend()
    # plt.savefig("figs/problem2/gamma_sensitivity.png")
    # plt.close()
    
    # print(f"Best gamma found: {best_gamma:.2f}")
    
    # # Analyze epsilon with best parameters so far
    # print("\nStarting epsilon analysis with best parameters...")
    # fixed_params['gamma'] = best_gamma  # Use best gamma found
    
    # # Test epsilon values
    # epsilons = np.linspace(0.05, 0.5, 10)  # Test reasonable exploration rates
    # epsilon_means, epsilon_stds = analyze_parameter_sensitivity(
    #     env, 'epsilon', epsilons, fixed_params
    # )
    
    # plt.figure(figsize=(12, 6))
    # plt.errorbar(epsilons, epsilon_means, yerr=epsilon_stds, fmt='o-', capsize=5)
    # plt.xlabel('Exploration Rate (ε)')
    # plt.ylabel('Average Total Reward')
    # plt.title(f'Epsilon Sensitivity Analysis (with α={best_alpha:.2e}, λ={best_lambda:.2f}, γ={best_gamma:.2f})')
    # plt.grid(True)
    # plt.fill_between(epsilons, epsilon_means - epsilon_stds, epsilon_means + epsilon_stds, alpha=0.2)
    
    # # Annotate the best epsilon
    # best_epsilon_idx = np.argmax(epsilon_means)
    # best_epsilon = epsilons[best_epsilon_idx]
    # plt.plot(best_epsilon, epsilon_means[best_epsilon_idx], 'r*', markersize=15, label=f'Best ε={best_epsilon:.2f}')
    # plt.legend()
    # plt.savefig("figs/problem2/epsilon_sensitivity.png")
    # plt.close()
    
    # print(f"Best epsilon found: {best_epsilon:.2f}")
    
    # # Final comparison with all optimized parameters
    # print("\nComparing learning curves with all optimized parameters...")
    # comparison_results = []
    
    # # 1. Initial parameters
    # rewards, _, _ = sarsa_lambda(
    #     env, episodes, alpha=0.1, gamma=1.0, lmbda=0.9, epsilon=0.2,
    #     order=order, max_steps=max_steps, momentum=momentum, nesterov=nesterov,
    #     decay_rate=decay_rate
    # )
    # comparison_results.append(('Initial Parameters', rewards))
    
    # # 2. Best alpha only
    # rewards, _, _ = sarsa_lambda(
    #     env, episodes, alpha=best_alpha, gamma=1.0, lmbda=0.9, epsilon=0.2,
    #     order=order, max_steps=max_steps, momentum=momentum, nesterov=nesterov,
    #     decay_rate=decay_rate
    # )
    # comparison_results.append(('Best α', rewards))
    
    # # 3. Best alpha and lambda
    # rewards, _, _ = sarsa_lambda(
    #     env, episodes, alpha=best_alpha, gamma=1.0, lmbda=best_lambda, epsilon=0.2,
    #     order=order, max_steps=max_steps, momentum=momentum, nesterov=nesterov,
    #     decay_rate=decay_rate
    # )
    # comparison_results.append(('Best α,λ', rewards))
    
    # # 4. Best alpha, lambda, and gamma
    # rewards, _, _ = sarsa_lambda(
    #     env, episodes, alpha=best_alpha, gamma=best_gamma, lmbda=best_lambda, epsilon=0.2,
    #     order=order, max_steps=max_steps, momentum=momentum, nesterov=nesterov,
    #     decay_rate=decay_rate
    # )
    # comparison_results.append(('Best α,λ,γ', rewards))
    
    # # 5. All best parameters
    # rewards, weights, indices = sarsa_lambda(
    #     env, episodes, alpha=best_alpha, gamma=best_gamma, lmbda=best_lambda, 
    #     epsilon=best_epsilon, order=order, max_steps=max_steps, momentum=momentum, 
    #     nesterov=nesterov, decay_rate=decay_rate
    # )
    # comparison_results.append(('All Best Parameters', rewards))
    
    # # Plot final comparison
    # plt.figure(figsize=(12, 6))
    # window = 50
    # for label, rewards in comparison_results:
    #     running_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    #     plt.plot(np.arange(len(running_avg)) + window-1, running_avg, label=label)
    
    # plt.xlabel("Episodes")
    # plt.ylabel("Average Total Reward")
    # plt.title("Learning Curves with Different Parameter Combinations")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("figs/problem2/final_parameter_comparison.png")
    # plt.close()
    
    # # (d2) Plot value function
    # positions = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 50)
    # velocities = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 50)
    # V_grid = np.zeros((50, 50))
    # Pi_grid = np.zeros((50, 50))
    
    # bounds = (env.observation_space.low, env.observation_space.high)
    # for i, pos in enumerate(positions):
    #     for j, vel in enumerate(velocities):
    #         state = np.array([pos, vel])
    #         features = fourier_basis(state, indices, bounds)
    #         Q_values = weights @ features
    #         V_grid[i, j] = np.max(Q_values)  # Value is max over actions
    #         Pi_grid[i, j] = np.argmax(Q_values)  # Policy is argmax over actions
    
    # # Plot value function
    # X, Y = np.meshgrid(positions, velocities)
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(X, Y, V_grid.T, cmap='viridis')
    # ax.set_xlabel('Position')
    # ax.set_ylabel('Velocity')
    # ax.set_zlabel('Value')
    # ax.set_title('Value Function of Optimal Policy')
    # plt.colorbar(surf)
    # plt.savefig("figs/problem2/value_function.png")
    # plt.close()
    
    # # (d3) Plot policy
    # fig = plt.figure(figsize=(10, 8))
    # plt.pcolormesh(X, Y, Pi_grid.T, cmap='viridis')
    # plt.colorbar(label='Action (0: Left, 1: No Push, 2: Right)')
    # plt.xlabel('Position')
    # plt.ylabel('Velocity')
    # plt.title('Optimal Policy')
    # plt.savefig("figs/problem2/policy.png")
    # plt.close()
    
    # (d4) Analysis with and without η = [0,0]
    # Train two models: one with and one without the constant basis
    results_with_constant = []
    results_without_constant = []
    
    for include_constant in [True, False]:
        # Create indices with or without [0,0]
        if include_constant:
            current_indices = indices
        else:
            # Remove the [0,0] index if it exists
            current_indices = indices[~np.all(indices == 0, axis=1)]
        
        # Train the model
        rewards, _, _ = sarsa_lambda(
            env, episodes, alpha, gamma, lmbda, epsilon, order, max_steps,
            momentum=momentum, nesterov=nesterov, decay_rate=decay_rate
        )
        

        if include_constant:
            results_with_constant = rewards
        else:
            results_without_constant = rewards
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    window = 50
    for rewards, label in [(results_with_constant, 'With η=[0,0]'), 
                          (results_without_constant, 'Without η=[0,0]')]:
        running_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(len(running_avg)) + window-1, running_avg, label=label)
    
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Effect of Including Constant Basis Function")
    plt.legend()
    plt.grid()
    plt.savefig("figs/problem2/constant_basis_comparison.png")
    plt.close()
    
