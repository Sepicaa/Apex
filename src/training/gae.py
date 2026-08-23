import jax
import jax.numpy as jnp
from src.training.rollout import Transition

def compute_gae(
    trajectories: Transition,
    last_val: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
):
    """Computes GAE advantages and target values backwards in time."""
    
    def _gae_step(gae_and_next_val, transition: Transition):
        last_gae, next_val = gae_and_next_val
        
        # 1. Mask out future values if the episode terminated
        non_terminal = 1.0 - transition.done
        
        # 2. TD Error: delta = r + gamma * V(s') * (1 - done) - V(s)
        delta = transition.reward + gamma * next_val * non_terminal - transition.value
        
        # 3. Recursive Advantage: A_t = delta + gamma * lambda * (1 - done) * A_{t+1}
        gae = delta + gamma * gae_lambda * non_terminal * last_gae
        
        # 4. Pass (current_gae, current_value) to previous timestep
        return (gae, transition.value), gae

    # Scan backwards over the collected rollout trajectory
    initial_carry = (jnp.zeros_like(last_val), last_val)
    _, advantages = jax.lax.scan(
        _gae_step,
        initial_carry,
        trajectories,
        reverse=True
    )
    
    # Target value for the Critic update: V_target = Advantage + V(s)
    targets = advantages + trajectories.value
    
    return advantages, targets