import torch

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_val: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
):
    """
    Computes GAE advantages and target values backwards in time.
    Expects inputs of shape (num_steps, num_envs).
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    next_val = last_val
    
    num_steps = rewards.shape[0]

    # Step backwards through the rollout trajectory
    for t in reversed(range(num_steps)):
        # 1. Mask out future values if the episode terminated at step t
        non_terminal = 1.0 - dones[t]
        
        # 2. TD Error: delta = r + gamma * V(s') * (1 - done) - V(s)
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        
        # 3. Recursive Advantage: A_t = delta + gamma * lambda * (1 - done) * A_{t+1}
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        
        advantages[t] = last_gae
        next_val = values[t]
        
    # Target value for the Critic update: V_target = Advantage + V(s)
    targets = advantages + values
    
    return advantages, targets