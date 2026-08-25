from typing import NamedTuple
import numpy as np
import torch
from src.training.networks import Actor, Critic


class RolloutBuffer(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    is_crashed: torch.Tensor


def collect_rollouts(
    envs,
    last_obs: np.ndarray,
    last_done: np.ndarray,
    actor: Actor,
    critic: Critic,
    num_steps: int,
    device: torch.device = torch.device("cpu")
):
    """
    Collects experience across all vectorized environments for `num_steps`.
    Preallocates PyTorch tensors for zero memory-allocation overhead during the rollout loop.
    """
    num_envs = envs.num_envs
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    # 1. Preallocate Rollout Storage Tensors
    obs_buffer = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
    actions_buffer = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32, device=device)
    rewards_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    log_probs_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    crashed_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    current_obs = last_obs
    current_done = last_done

    for step in range(num_steps):
        # Convert observation and done flag to tensors
        obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(current_done, dtype=torch.float32, device=device)

        # 2. Policy & Value Network Forward Pass (No gradients during rollouts)
        with torch.no_grad():
            action_tensor, log_prob_tensor, _ = actor.get_action(obs_tensor)
            value_tensor = critic(obs_tensor)

        # 3. Environment Step
        # Clip action to valid range and pass to C++ MuJoCo via NumPy
        action_np = action_tensor.cpu().numpy()
        action_clipped = np.clip(action_np, -1.0, 1.0)

        next_obs, rewards, terminated, truncated, infos = envs.step(action_clipped)
        dones = np.logical_or(terminated, truncated).astype(np.float32)

        # Extract crash flags from info dict
        if "is_crashed" in infos:
            is_crashed = np.array(infos["is_crashed"], dtype=np.float32)
        else:
            is_crashed = dones.copy()

        # 4. Save into Preallocated Buffer
        obs_buffer[step] = obs_tensor
        actions_buffer[step] = action_tensor
        values_buffer[step] = value_tensor
        log_probs_buffer[step] = log_prob_tensor
        rewards_buffer[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones_buffer[step] = done_tensor
        crashed_buffer[step] = torch.as_tensor(is_crashed, dtype=torch.float32, device=device)

        # 5. Advance State
        current_obs = next_obs
        current_done = dones

    buffer = RolloutBuffer(
        obs=obs_buffer,
        actions=actions_buffer,
        rewards=rewards_buffer,
        dones=dones_buffer,
        values=values_buffer,
        log_probs=log_probs_buffer,
        is_crashed=crashed_buffer,
    )

    return current_obs, current_done, buffer