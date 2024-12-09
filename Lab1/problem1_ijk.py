# Bjarni Haukur Bjarnason (bhbj@kth.se)
# Seán O Riordan (seanor@kth.se)

import random
import argparse
from pathlib import Path
from enum import IntEnum
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

Path(__file__).parent.parent.joinpath('figs').mkdir(parents=True, exist_ok=True)

class State(IntEnum):
    EMPTY = 0
    WALL = 1
    KEY = 2    # Position C
    GOAL = 3   # Position B

class Reward(IntEnum):
    WIN = 1000
    KEY = 100
    EATEN = -100
    POISONED = 0  # penalizing steps is already minimizing the number of steps, and therefore the likelihood of being poisoned
    STEP = -1

class ModifiedMaze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    
    def __init__(self, poison_prob=1/50):
        self.poison_prob = poison_prob
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
    
    def step(self, state, action):
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
        if random.random() < self.poison_prob:
            return None, Reward.POISONED, True
            
        step_penalty = Reward.STEP
        
        return (new_player_pos, new_minotaur_pos, new_has_key), step_penalty, False

class QLearningAgent:
    def __init__(self, env: ModifiedMaze, epsilon: float, alpha_power: float = 0.8, 
                 epsilon_decay: float = 1.0, epsilon_min: float = 0.01):
        self.env = env
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
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
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
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

def train_agent(agent, n_episodes: int, gamma: float = 0.98, max_steps: int = 100) -> Tuple[List[float], List[float]]:
    initial_values = []
    running_rewards = []  # Track running average of rewards
    episode_rewards = []  # Track individual episode rewards
    episode_steps = []   # Track steps per episode
    
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
            
        episode_steps.append(step)
        initial_state_key = str(((0, 0), (6, 5), False))
        initial_values.append(np.max(agent.Q[initial_state_key]))
        running_rewards.append(total_reward)
        episode_rewards.append(total_reward)
        
        # Decay epsilon after each episode
        agent.decay_epsilon()
        
        if episode % 1000 == 0:
            avg_reward = np.mean(running_rewards[-1000:]) if running_rewards else 0
            avg_steps = np.mean(episode_steps[-1000:]) if episode_steps else 0
            print(
                f"Episode {episode}/{n_episodes},",
                f"Initial State Value: {initial_values[-1]:.3f},",
                f"Avg Reward: {avg_reward:.3f},",
                f"Avg Steps: {avg_steps:.1f},",
                f"Epsilon: {agent.epsilon:.3f}",
            )
            
    return initial_values, episode_rewards

def evaluate_policy(agent, n_episodes: int = 1000, max_steps: int = 200) -> dict:
    outcomes = {
        'Win': 0,
        'Eaten': 0,
        'Poisoned': 0,
        'Timeout': 0
    }
    
    for episode in range(n_episodes):
        state = ((0, 0), (6, 5), False)
        steps = 0
        
        while steps < max_steps:
            action = agent.get_action(state, training=False)
            next_state, reward, done = agent.env.step(state, action)
            
            if done:
                # hacky
                if reward == Reward.WIN:
                    outcomes['Win'] += 1
                elif reward == Reward.EATEN:
                    outcomes['Eaten'] += 1
                elif reward == Reward.POISONED:
                    outcomes['Poisoned'] += 1
                break
                
            state = next_state
            steps += 1
            
        if steps >= max_steps:
            outcomes['Timeout'] += 1
    
    return outcomes

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate RL agents on the Modified Maze environment')
    
    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=50000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount factor')
    
    # Agent parameters
    parser.add_argument('--epsilon', type=float, default=0.1, help='Initial epsilon for epsilon-greedy exploration')
    parser.add_argument('--epsilon2', type=float, default=0.2, help='Initial epsilon for second agent')
    parser.add_argument('--epsilon_decay', type=float, default=1.0, help='Epsilon decay rate (1.0 means no decay)')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum epsilon value')
    parser.add_argument('--alpha_power', type=float, default=2/3, help='Power for learning rate decay')
    
    # Plotting parameters
    parser.add_argument('--window', type=int, default=500, help='Window size for moving average in plots')
    
    # Environment parameters
    parser.add_argument('--poison_prob', type=float, default=1/50, help='Probability of poisoning')
    
    args = parser.parse_args()
    
    # Create environment and agents
    env = ModifiedMaze(poison_prob=args.poison_prob)
    
    # Train Q-learning agents
    print("Training Q-learning agents...")
    q_agent1 = QLearningAgent(env, epsilon=args.epsilon, alpha_power=args.alpha_power,
                             epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min)
    q_agent2 = QLearningAgent(env, epsilon=args.epsilon2, alpha_power=args.alpha_power,
                             epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min)
    
    q_values1, q_rewards1 = train_agent(q_agent1, n_episodes=args.n_episodes, 
                                      gamma=args.gamma, max_steps=args.max_steps)
    q_values2, q_rewards2 = train_agent(q_agent2, n_episodes=args.n_episodes, 
                                      gamma=args.gamma, max_steps=args.max_steps)
    
    # Plot Q-learning results
    plt.figure(figsize=(10, 6))
    
    # Plot moving averages for rewards
    ma1 = np.convolve(q_rewards1, np.ones(args.window)/args.window, mode='valid')
    ma2 = np.convolve(q_rewards2, np.ones(args.window)/args.window, mode='valid')
    plt.plot(np.arange(len(ma1)) + args.window//2, ma1, color='blue', 
            label=f'ε={args.epsilon} (moving avg {args.window} over episodes)')
    plt.plot(np.arange(len(ma2)) + args.window//2, ma2, color='orange', 
            label=f'ε={args.epsilon2} (moving avg {args.window} over episodes)')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(
        'Q-Learning Training Progress: Total Reward per Episode' + \
        f' (epsilon_decay={args.epsilon_decay}, min_epsilon={args.epsilon_min})' if args.epsilon_decay != 1.0 else ''
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figs/q_learning_rewards.png')
    plt.close()
    
    # Plot Q-learning convergence
    plt.figure(figsize=(10, 6))
    plt.plot(q_values1, color='blue', label=f'ε={args.epsilon}')
    plt.plot(q_values2, color='orange', label=f'ε={args.epsilon2}')
    
    plt.xlabel('Episode')
    plt.ylabel('Initial State Value')
    plt.title('Q-Learning Convergence' + \
              f' (epsilon_decay={args.epsilon_decay}, min_epsilon={args.epsilon_min})' if args.epsilon_decay != 1.0 else '')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/q_learning_convergence.png')
    plt.close()
    
    # Train SARSA agents
    print("\nTraining SARSA agents...")
    sarsa_agent1 = SarsaAgent(env, epsilon=args.epsilon, alpha_power=args.alpha_power,
                             epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min)
    sarsa_agent2 = SarsaAgent(env, epsilon=args.epsilon2, alpha_power=args.alpha_power,
                             epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min)
    
    sarsa_values1, sarsa_rewards1 = train_agent(sarsa_agent1, n_episodes=args.n_episodes,
                                               gamma=args.gamma, max_steps=args.max_steps)
    sarsa_values2, sarsa_rewards2 = train_agent(sarsa_agent2, n_episodes=args.n_episodes,
                                               gamma=args.gamma, max_steps=args.max_steps)
    
    # Plot SARSA results
    plt.figure(figsize=(10, 6))
    ma1 = np.convolve(sarsa_rewards1, np.ones(args.window)/args.window, mode='valid')
    ma2 = np.convolve(sarsa_rewards2, np.ones(args.window)/args.window, mode='valid')
    plt.plot(np.arange(len(ma1)) + args.window//2, ma1, color='blue', 
            label=f'ε={args.epsilon} (moving avg)')
    plt.plot(np.arange(len(ma2)) + args.window//2, ma2, color='orange', 
            label=f'ε={args.epsilon2} (moving avg)')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SARSA Training Progress: Total Reward per Episode' + \
              f' (epsilon_decay={args.epsilon_decay}, min_epsilon={args.epsilon_min})' if args.epsilon_decay != 1.0 else '')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figs/sarsa_rewards.png')
    plt.close()
    
    # Plot SARSA convergence
    plt.figure(figsize=(10, 6))
    plt.plot(sarsa_values1, color='blue', label=f'ε={args.epsilon}')
    plt.plot(sarsa_values2, color='orange', label=f'ε={args.epsilon2}')
    
    plt.xlabel('Episode')
    plt.ylabel('Initial State Value')
    plt.title('SARSA Convergence' + \
              f' (epsilon_decay={args.epsilon_decay}, min_epsilon={args.epsilon_min})' if args.epsilon_decay != 1.0 else '')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/sarsa_convergence.png')
    plt.close()
    
    # Evaluate final policies
    print("\nEvaluating final policies...")
    q_outcomes1 = evaluate_policy(q_agent1)
    q_outcomes2 = evaluate_policy(q_agent2)
    sarsa_outcomes1 = evaluate_policy(sarsa_agent1)
    sarsa_outcomes2 = evaluate_policy(sarsa_agent2)
    
    # Plot outcomes for Q-learning agents
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    total_outcomes = sum(q_outcomes1.values())
    outcomes_percentages = {k: (v/total_outcomes)*100 for k, v in q_outcomes1.items()}
    plt.bar(outcomes_percentages.keys(), outcomes_percentages.values())
    plt.title(f'Q-Learning Outcomes (ε={args.epsilon})')
    plt.ylabel('Percentage of Episodes (%)')
    
    plt.subplot(1, 2, 2)
    total_outcomes = sum(q_outcomes2.values())
    outcomes_percentages = {k: (v/total_outcomes)*100 for k, v in q_outcomes2.items()}
    plt.bar(outcomes_percentages.keys(), outcomes_percentages.values())
    plt.title(f'Q-Learning Outcomes (ε={args.epsilon2})')
    plt.ylabel('Percentage of Episodes (%)')
    plt.tight_layout()
    plt.savefig('figs/q_learning_outcomes.png')
    plt.close()
    
    # Plot outcomes for SARSA agents
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    total_outcomes = sum(sarsa_outcomes1.values())
    outcomes_percentages = {k: (v/total_outcomes)*100 for k, v in sarsa_outcomes1.items()}
    plt.bar(outcomes_percentages.keys(), outcomes_percentages.values())
    plt.title(f'SARSA Outcomes (ε={args.epsilon})')
    plt.ylabel('Percentage of Episodes (%)')
    
    plt.subplot(1, 2, 2)
    total_outcomes = sum(sarsa_outcomes2.values())
    outcomes_percentages = {k: (v/total_outcomes)*100 for k, v in sarsa_outcomes2.items()}
    plt.bar(outcomes_percentages.keys(), outcomes_percentages.values())
    plt.title(f'SARSA Outcomes (ε={args.epsilon2})')
    plt.ylabel('Percentage of Episodes (%)')
    plt.tight_layout()
    plt.savefig('figs/sarsa_outcomes.png')
    plt.close()
