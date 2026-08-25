import torch
from src.training.rollout import Transition

def compute_gae(
    trajectories: Transition,
    last_val: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
):
    """
    Computes GAE advantages and target values backwards in time.
    Expects trajectories to contain tensors of shape (num_steps, num_envs).
    """
    # Preallocate the advantage tensor on the same device as the rewards
    advantages = torch.zeros_like(trajectories.rewards)
    
    last_gae = 0.0
    next_val = last_val
    
    num_steps = trajectories.rewards.shape[0]

    # Step backwards through the rollout trajectory
    for t in reversed(range(num_steps)):
        # 1. Mask out future values if the episode terminated at step t
        non_terminal = 1.0 - trajectories.dones[t]
        
        # 2. TD Error: delta = r + gamma * V(s') * (1 - done) - V(s)
        delta = trajectories.rewards[t] + gamma * next_val * non_terminal - trajectories.values[t]
        
        # 3. Recursive Advantage: A_t = delta + gamma * lambda * (1 - done) * A_{t+1}
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        
        # 4. Store current advantage and update next_val for the previous timestep
        advantages[t] = last_gae
        next_val = trajectories.values[t]
        
    # Target value for the Critic update: V_target = Advantage + V(s)
    targets = advantages + trajectories.values
    
    return advantages, targets