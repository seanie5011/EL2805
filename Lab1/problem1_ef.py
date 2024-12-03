import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, value_iteration, animate_solution


maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

# Problem 1e: Solve with geometric lifetime
print("Solving with geometric lifetime...")
env = Maze(maze)
mean_lifetime = 30

# Gamma models survival probability per step, ensures future rewards are discounted appropriately for a finite lifetime 
gamma = 1 - 1/mean_lifetime  

epsilon = 1e-3  # Convergence threshold
max_iter = 5000

V, policy = value_iteration(env, gamma, epsilon, max_iter)

# Problem 1f: Simulate 10,000 games
print("\nSimulating 10,000 games...")
n_simulations = 10000
start = ((0,0), (6,5))
wins = 0

for i in range(n_simulations):
    path = env.simulate(start, policy, 'ValIter', horizon=100, poison_prob=1/30)[0]
    if path[-1] == 'Win':
        wins += 1

success_probability = wins / n_simulations
print(f"\nSuccess probability over {n_simulations} simulations: {success_probability:.3f}")