import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    """Initializes linear layers with orthogonal weights to match the JAX implementation."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super(Actor, self).__init__()
        
        # Hidden layers with sqrt(2) scaling
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        
        # Output layer with 0.01 scaling
        self.mean_layer = layer_init(nn.Linear(256, action_dim), std=0.01)
        
        # Learnable standard deviation parameter (initialized to zeros)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        hidden = self.net(x)
        mean = self.mean_layer(hidden)
        # Expand log_std to match the batch dimension of the mean
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std
    
    def get_distribution(self, x):
        """Helper method to return a PyTorch Normal distribution."""
        mean, std = self.forward(x)
        return Normal(mean, std)

    def get_action(self, x, action=None):
        """Samples an action and computes its log probability and entropy for PPO."""
        dist = self.get_distribution(x)
        
        if action is None:
            action = dist.sample()
            
        # Sum log probabilities across the action dimensions
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super(Critic, self).__init__()
        
        # Hidden layers with sqrt(2) scaling
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        
        # Critic output with standard 1.0 scaling
        self.value_layer = layer_init(nn.Linear(256, 1), std=1.0)

    def forward(self, x):
        hidden = self.net(x)
        value = self.value_layer(hidden)
        # Squeeze the final dimension to output a scalar per batch item
        return value.squeeze(-1)