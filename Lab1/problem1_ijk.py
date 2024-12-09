# Bjarni Haukur Bjarnason (bhbj@kth.se)
# Seán O Riordan (seanor@kth.se)

import random
from enum import IntEnum, auto
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

class State(IntEnum):
    EMPTY = 0
    WALL = 1
    KEY = 2    # Position C
    GOAL = 3   # Position B

class Reward(IntEnum):
    WIN = 1000
    KEY = 100
    EATEN = -100
    POISONED = -10  # New negative reward for being poisoned
    STEP = -1

class ModifiedMaze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    
    def __init__(self):
        # Convert maze to numeric format for easier handling
        self.maze = np.array([
            [0, 0, 1, 0, 0, 0, 0, 2],  # 2 represents key position (C)
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 3, 0, 0]   # 3 represents goal position (B)
        ])
        self.actions = {
            self.STAY: (0, 0),
            self.MOVE_LEFT: (0, -1),
            self.MOVE_RIGHT: (0, 1),
            self.MOVE_UP: (-1, 0),
            self.MOVE_DOWN: (1, 0)
        }
    
    def is_valid_position(self, pos):
        """Check if a position is valid (within bounds and not a wall)."""
        row, col = pos
        if 0 <= row < self.maze.shape[0] and 0 <= col < self.maze.shape[1]:
            return self.maze[row, col] != 1
        return False
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_valid_moves(self, position):
        """Get valid moves from a position."""
        valid_moves = []
        for action, (dy, dx) in self.actions.items():
            new_y = position[0] + dy
            new_x = position[1] + dx
            new_pos = (new_y, new_x)
            if self.is_valid_position(new_pos):
                valid_moves.append((action, new_pos))
        return valid_moves
    
    def get_minotaur_moves(self, minotaur_pos, player_pos):
        """Get next minotaur position based on strategy."""
        valid_moves = self.get_valid_moves(minotaur_pos)
        if not valid_moves:
            return minotaur_pos
            
        if random.random() < 0.35:  # Strategic movement
            current_dist = self.manhattan_distance(minotaur_pos, player_pos)
            strategic_moves = []
            for _, next_pos in valid_moves:
                dist = self.manhattan_distance(next_pos, player_pos)
                if dist < current_dist:
                    strategic_moves.append(next_pos)
            if strategic_moves:
                return random.choice(strategic_moves)
        
        # Random movement among valid moves
        return random.choice([pos for _, pos in valid_moves])
    
    def step(self, state, action, poison_prob=1/30):
        """Take a step in the environment."""
        player_pos, minotaur_pos, has_key = state
        
        # Get player's next position
        dy, dx = self.actions[action]
        new_player_y = player_pos[0] + dy
        new_player_x = player_pos[1] + dx
        new_player_pos = (new_player_y, new_player_x)
        
        # Validate move
        if not self.is_valid_position(new_player_pos):
            new_player_pos = player_pos
        
        # Update key status
        new_has_key = has_key
        if self.maze[new_player_pos] == State.KEY:
            new_has_key = True
        
        # Move minotaur
        new_minotaur_pos = self.get_minotaur_moves(minotaur_pos, new_player_pos)
        
        # Check win/lose conditions
        if new_player_pos == new_minotaur_pos:
            return None, Reward.EATEN, True
            
        if self.maze[new_player_pos] == State.GOAL:
            if new_has_key:
                return None, Reward.WIN, True
            else:
                return (new_player_pos, new_minotaur_pos, new_has_key), Reward.STEP, False
                
        if self.maze[new_player_pos] == State.KEY and not has_key:
            return (new_player_pos, new_minotaur_pos, new_has_key), Reward.KEY, False
            
        # Check for poison
        if random.random() < poison_prob:
            return None, Reward.POISONED, True
            
        step_penalty = Reward.STEP
        
        return (new_player_pos, new_minotaur_pos, new_has_key), step_penalty, False

class QLearningAgent:
    def __init__(self, env: ModifiedMaze, epsilon: float, alpha_power: float = 0.8):
        self.env = env
        self.epsilon = epsilon
        self.alpha_power = alpha_power
        self.Q = {}
        self.visit_counts = {}
        
    def get_action(self, state, training: bool = True) -> int:
        state_key = str(state)
        if state_key not in self.Q:
            self.Q[state_key] = np.random.randn(5) * 0.1
            self.visit_counts[state_key] = np.zeros(5)
            
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, 4)
        return np.argmax(self.Q[state_key])
    
    def update(self, state, action: int, next_state, reward: float, gamma: float):
        state_key = str(state)
        next_state_key = str(next_state) if next_state is not None else None
        
        if state_key not in self.Q:
            self.Q[state_key] = np.random.randn(5) * 0.1
            self.visit_counts[state_key] = np.zeros(5)
            
        self.visit_counts[state_key][action] += 1
        alpha = min(0.1, 1 / (self.visit_counts[state_key][action] ** self.alpha_power))  # Cap learning rate
        
        if next_state is None:  # Terminal state
            target = reward
        else:
            if next_state_key not in self.Q:
                self.Q[next_state_key] = np.random.randn(5) * 0.1
                self.visit_counts[next_state_key] = np.zeros(5)
            target = reward + gamma * np.max(self.Q[next_state_key])
            
        self.Q[state_key][action] += alpha * (target - self.Q[state_key][action])

class SarsaAgent(QLearningAgent):
    def update(self, state, action: int, next_state, next_action: int, reward: float, gamma: float):
        state_key = str(state)
        next_state_key = str(next_state) if next_state is not None else None
        
        if state_key not in self.Q:
            self.Q[state_key] = np.random.randn(5) * 0.1
            self.visit_counts[state_key] = np.zeros(5)
            
        self.visit_counts[state_key][action] += 1
        alpha = min(0.1, 1 / (self.visit_counts[state_key][action] ** self.alpha_power))  # Cap learning rate
        
        if next_state is None:  # Terminal state
            target = reward
        else:
            if next_state_key not in self.Q:
                self.Q[next_state_key] = np.ones(5) * 0.1
                self.visit_counts[next_state_key] = np.zeros(5)
            target = reward + gamma * self.Q[next_state_key][next_action]
            
        self.Q[state_key][action] += alpha * (target - self.Q[state_key][action])

def train_agent(agent, n_episodes: int, gamma: float = 0.98, max_steps: int = 100) -> List[float]:
    initial_values = []
    running_rewards = []  # Track running average of rewards
    
    for episode in range(n_episodes):
        state = ((0, 0), (6, 5), False)
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            
            if isinstance(agent, SarsaAgent):
                next_state, reward, done = agent.env.step(state, action)
                next_action = agent.get_action(next_state) if next_state is not None else None
                agent.update(state, action, next_state, next_action, reward, gamma)
                state = next_state
                action = next_action
            else:  # Q-learning
                next_state, reward, done = agent.env.step(state, action)
                agent.update(state, action, next_state, reward, gamma)
                state = next_state
            
            total_reward += reward
            if done or state is None:
                break
                
        initial_state_key = str(((0, 0), (6, 5), False))
        initial_values.append(np.max(agent.Q[initial_state_key]))
        running_rewards.append(total_reward)
        
        if episode % 1000 == 0:
            avg_reward = np.mean(running_rewards[-1000:]) if running_rewards else 0
            print(
                f"Episode {episode}/{n_episodes},",
                f"Initial State Value: {initial_values[-1]:.3f},",
                f"Avg Reward: {avg_reward:.3f}",
            )
            
    return initial_values

def evaluate_policy(agent, n_episodes: int = 1000, max_steps: int = 200) -> float:
    wins = 0
    
    for episode in range(n_episodes):
        state = ((0, 0), (6, 5), False)
        steps = 0
        
        while steps < max_steps:
            action = agent.get_action(state, training=False)
            next_state, reward, done = agent.env.step(state, action)
            
            if done:
                if reward == Reward.WIN:
                    wins += 1
                break
                
            state = next_state
            steps += 1
                
    return wins / n_episodes

def visualize_path(maze: np.ndarray, path: List[Tuple], save_path: str = 'path_visualization.png'):
    plt.figure(figsize=(10, 10))
    
    # Create a visual representation of the maze
    maze_visual = np.copy(maze)
    plt.imshow(maze == State.WALL, cmap='binary')
    
    # Add walls as black rectangles
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == State.WALL:
                plt.gca().add_patch(Rectangle((j-0.5, i-0.5), 1, 1, color='black'))
    
    # Mark special positions
    plt.text(0, 0, 'A', ha='center', va='center', color='blue', fontsize=12, fontweight='bold')
    plt.text(7, 0, 'C', ha='center', va='center', color='blue', fontsize=12, fontweight='bold')
    plt.text(5, 6, 'B', ha='center', va='center', color='blue', fontsize=12, fontweight='bold')
    
    # Create color gradient from red to green
    colors = [(1, 0, 0), (0, 1, 0)]  # Red to Green
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Filter out terminal states and extract player positions
    valid_states = [p for p in path if isinstance(p, tuple)]
    player_positions = [(p[0][1], p[0][0]) for p in valid_states]  # Convert to (x,y) for plotting
    
    if len(player_positions) > 1:
        points = np.array(player_positions)
        segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)
        dists = np.linspace(0, 1, len(segments))
        
        for i, segment in enumerate(segments):
            plt.plot(segment[:, 0], segment[:, 1], color=cmap(dists[i]), linewidth=2, zorder=2)
        
        plt.plot(player_positions[0][0], player_positions[0][1], 'ro', label='Start', markersize=10, zorder=3)
        plt.plot(player_positions[-1][0], player_positions[-1][1], 'go', label='End', markersize=10, zorder=3)
    
    plt.grid(True)
    plt.title(f"{'Sarsa' if 'sarsa' in save_path else 'Q-learning'} Path Through Maze")
    plt.legend()
    plt.savefig(f"figs/{save_path}")
    plt.close()

if __name__ == "__main__":
    env = ModifiedMaze()
    
    # Train with more conservative parameters
    print("\nTraining Q-learning agents...")
    q_agent1 = QLearningAgent(env, epsilon=0.1, alpha_power=2/3)
    q_agent2 = QLearningAgent(env, epsilon=0.2, alpha_power=2/3)
    
    q_values1 = train_agent(q_agent1, n_episodes=50000)
    q_values2 = train_agent(q_agent2, n_episodes=50000)
    
    # Plot Q-learning results with both raw and moving average
    plt.figure(figsize=(10, 6))
    window = 500
    
    
    # Plot moving averages
    ma1 = np.convolve(q_values1, np.ones(window)/window, mode='valid')
    ma2 = np.convolve(q_values2, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(len(ma1)) + window//2, ma1, color='blue', label='ε=0.1 (moving avg)')
    plt.plot(np.arange(len(ma2)) + window//2, ma2, color='orange', label='ε=0.2 (moving avg)')
    
    plt.xlabel('Episode')
    plt.ylabel('Initial State Value')
    plt.title('Q-learning Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/q_learning_convergence.png')
    plt.close()
    
    print("\nTraining SARSA agents...")
    sarsa_agent1 = SarsaAgent(env, epsilon=0.1, alpha_power=2/3)
    sarsa_agent2 = SarsaAgent(env, epsilon=0.2, alpha_power=2/3)
    
    sarsa_values1 = train_agent(sarsa_agent1, n_episodes=50000)
    sarsa_values2 = train_agent(sarsa_agent2, n_episodes=50000)
    
    # Plot SARSA results with both raw and moving average
    plt.figure(figsize=(10, 6))
    
    # Plot moving averages
    ma1 = np.convolve(sarsa_values1, np.ones(window)/window, mode='valid')
    ma2 = np.convolve(sarsa_values2, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(len(ma1)) + window//2, ma1, color='blue', label='ε=0.1 (moving avg)')
    plt.plot(np.arange(len(ma2)) + window//2, ma2, color='orange', label='ε=0.2 (moving avg)')
    
    plt.xlabel('Episode')
    plt.ylabel('Initial State Value')
    plt.title('SARSA Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/sarsa_convergence.png')
    plt.close()
    
    # Evaluate final policies
    print("\nEvaluating policies...")
    q_success = evaluate_policy(q_agent1)
    sarsa_success = evaluate_policy(sarsa_agent1)
    
    print(f"\nQ-learning success probability: {q_success:.3f}")
    print(f"SARSA success probability: {sarsa_success:.3f}")
    
    # Visualize Q-learning path
    print("\nSimulating one episode with trained Q-learning agent...")
    state = ((0, 0), (6, 5), False)
    q_path = []
    total_reward = 0
    
    while True:
        q_path.append(state)
        action = q_agent1.get_action(state, training=False)
        next_state, reward, done = env.step(state, action)
        total_reward += reward
        
        if done:
            q_path.append(next_state if next_state is not None else 'Terminal')
            break
            
        state = next_state
    
    print(f"Q-learning episode finished with total reward: {total_reward}")
    visualize_path(env.maze, q_path, 'q_learning_path.png')
    
    # Visualize SARSA path
    print("\nSimulating one episode with trained SARSA agent...")
    state = ((0, 0), (6, 5), False)
    sarsa_path = []
    total_reward = 0
    
    while True:
        sarsa_path.append(state)
        action = sarsa_agent1.get_action(state, training=False)
        next_state, reward, done = env.step(state, action)
        total_reward += reward
        
        if done:
            sarsa_path.append(next_state if next_state is not None else 'Terminal')
            break
            
        state = next_state
    
    print(f"SARSA episode finished with total reward: {total_reward}")
    visualize_path(env.maze, sarsa_path, 'sarsa_path.png')
