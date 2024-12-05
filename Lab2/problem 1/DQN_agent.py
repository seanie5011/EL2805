# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch
import torch.nn as nn

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action

class DQNAgent(nn.Module):
    '''
    Deep Q-Network agent.

    Args:
        n_actions (int): number of actions

    Attributes:
        n_actions (int): where we store the number of actions
        last_action (int): last action taken by the agent
    '''

    def __init__(self, n_actions: int, input_size: int, output_size: int):
        super().__init__()
        self.n_actions = n_actions
        self.last_action = None

        self.hidden_neurons = 32

        self.input_layer = nn.Linear(input_size, self.hidden_neurons)  # First layer: state -> hidden layer
        self.hidden_layer = nn.Linear(self.hidden_neurons, self.hidden_neurons)  # Second layer: hidden -> hidden layer
        self.output_layer = nn.Linear(self.hidden_neurons, output_size)  # Output layer: hidden -> Q-values
        self.activation = nn.ReLU()  # ReLU activation function for hidden layers

    def forward(self, x: np.ndarray):
        ''' Performs a forward computation '''
        x = self.activation(self.input_layer(x))  # Apply input layer and ReLU
        x = self.activation(self.hidden_layer(x))  # Apply hidden layer and ReLU
        return self.output_layer(x)  # Return Q-values for all actions
