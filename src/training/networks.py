import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np # For sqrt

class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Hidden layers with sqrt(2) scaling
        init_hidden = nn.initializers.orthogonal(np.sqrt(2.0))
        
        x = nn.Dense(512, kernel_init=init_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=init_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=init_hidden)(x)
        x = nn.tanh(x)
        
        # Output layer with 0.01 scaling
        mean = nn.Dense(
            self.action_dim, 
            kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.clip(log_std, -100.0, 10.0)
        return mean, log_std

class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        init_hidden = nn.initializers.orthogonal(np.sqrt(2.0))
        
        x = nn.Dense(512, kernel_init=init_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=init_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=init_hidden)(x)
        x = nn.tanh(x)
        
        # Critic output with standard 1.0 scaling
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return jnp.squeeze(value, axis=-1)
