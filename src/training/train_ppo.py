import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, SubsetRandomSampler
from typing import NamedTuple, Tuple, Dict
import numpy as np

from src.training.networks import Actor, Critic
from src.training.rollout import collect_rollouts, Transition
from src.training.gae import compute_gae

class PPOConfig(NamedTuple):
    num_envs: int = 128
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.05
    num_steps: int = 128
    num_epochs: int = 5
    num_minibatches: int = 8
    max_grad_norm: float = 0.5


# --- 1. TrainState Initializer Helper ---

def create_train_states(
    obs_dim: int, action_dim: int, cfg: PPOConfig, device: torch.device
) -> Tuple[Actor, Critic, optim.Optimizer, optim.Optimizer]:
    
    # Initialize Networks
    actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(device)
    critic = Critic(obs_dim=obs_dim).to(device)

    # Initialize Optimizers (replaces Flax TrainState)[cite: 5]
    actor_optim = optim.AdamW(actor.parameters(), lr=cfg.lr_actor, weight_decay=cfg.weight_decay)
    critic_optim = optim.AdamW(critic.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay)

    return actor, critic, actor_optim, critic_optim


# --- 2. Multi-Epoch Minibatch Update (Combines JAX Loss & Epoch functions) ---

def update_ppo(
    actor: Actor, critic: Critic,
    actor_optim: optim.Optimizer, critic_optim: optim.Optimizer,
    buffer: Transition, advantages: torch.Tensor, targets: torch.Tensor,
    cfg: PPOConfig
) -> Dict[str, float]:
    
    # --- Flatten the Rollout Buffer (num_steps, num_envs) -> (num_steps * num_envs) ---
    b_obs = buffer.obs.view(-1, buffer.obs.shape[-1])
    b_actions = buffer.actions.view(-1, buffer.actions.shape[-1])
    b_log_probs = buffer.log_probs.view(-1)
    
    b_advantages = advantages.view(-1)
    b_targets = targets.view(-1)
    
    batch_size = b_obs.shape[0]
    minibatch_size = batch_size // cfg.num_minibatches
    
    # Advantage Normalization
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    
    total_p_loss, total_v_loss, total_ent = 0.0, 0.0, 0.0
    
    # --- Multi-Epoch Minibatch Optimization Loop ---
    for epoch in range(cfg.num_epochs):
        # Create random permutations for minibatches
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=True)
        
        for indices in sampler:
            mb_obs = b_obs[indices]
            mb_actions = b_actions[indices]
            mb_log_probs = b_log_probs[indices]
            mb_advantages = b_advantages[indices]
            mb_targets = b_targets[indices]
            
            # --- ACTOR UPDATE ---
            _, new_log_probs, entropy = actor.get_action(mb_obs, mb_actions)
            
            # Probability ratio: r(theta) = exp(log_pi - log_pi_old)
            ratio = torch.exp(new_log_probs - mb_log_probs)
        
            # Clipped surrogate objective
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            entropy_loss = entropy.mean()
            total_actor_loss = policy_loss - cfg.entropy_coef * entropy_loss
            
            # Backpropagation
            actor_optim.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_optim.step()
            
            # --- CRITIC UPDATE ---
            new_values = critic(mb_obs)
            value_loss = 0.5 * ((new_values - mb_targets) ** 2).mean()
            
            # Backpropagation
            critic_optim.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_optim.step()
            
            # Track metrics
            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            total_ent += entropy_loss.item()
            
    # Compute averages across all updates
    num_updates = cfg.num_epochs * cfg.num_minibatches
    return {
        "policy_loss": total_p_loss / num_updates,
        "value_loss": total_v_loss / num_updates,
        "entropy": total_ent / num_updates
    }


# --- 3. Single PPO Iteration Step ---

def ppo_train_iteration(
    envs, current_obs: np.ndarray, current_done: np.ndarray,
    actor: Actor, critic: Critic,
    actor_optim: optim.Optimizer, critic_optim: optim.Optimizer,
    cfg: PPOConfig, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    
    # Collect rollouts
    next_obs, trajectories = collect_rollouts(
        envs, current_obs, actor, critic, cfg.num_steps, device
    )

    # Value estimate for the very last step for GAE bootstrap
    with torch.no_grad():
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        last_val = critic(next_obs_tensor)

    # Compute GAE advantages & target returns
    advantages, targets = compute_gae(
        trajectories, last_val, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda
    )

    # Run multi-epoch minibatch PPO updates
    metrics = update_ppo(
        actor, critic, actor_optim, critic_optim,
        trajectories, advantages, targets, cfg
    )
    
    # Export full trajectories to Python side for rolling episodic tracking in main.py
    metrics["step_rewards"] = trajectories.rewards.cpu().numpy()
    metrics["step_dones"] = trajectories.dones.cpu().numpy()
    metrics["step_crashes"] = trajectories.is_crashed.cpu().numpy()
    
    # Pass along the latest done flag array
    next_done = trajectories.dones[-1].cpu().numpy()
    
    return next_obs, next_done, metrics