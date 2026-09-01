import jax
import jax.numpy as jnp
from typing import NamedTuple

from src.training.networks import Actor, Critic  # Fixed import path!

class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    log_prob: jax.Array
    is_crashed: jax.Array

def collect_rollouts(env, env_state, actor, critic, actor_params, critic_params, rng, num_steps, global_iteration):
    
    def _step(runner_state, _):
        current_env_state, last_obs, key = runner_state
        
        # 1. Forward Pass & Action Sampling
        key, action_key = jax.random.split(key)
        mean, log_std = actor.apply(actor_params, last_obs)
        std = jnp.exp(log_std)
        
        # Sample raw action and clip for the environment
        raw_action = mean + std * jax.random.normal(action_key, mean.shape)
        action_clipped = jnp.clip(raw_action, -1.0, 1.0)
        
        # 2. Value Prediction & Log Probability
        value = critic.apply(critic_params, last_obs)
        log_prob = -0.5 * jnp.sum(
            ((raw_action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), 
            axis=-1
        )
        
        # 3. Environment Step (Progress logic completely removed)
        next_env_state = env.step(current_env_state, action_clipped)
        is_crashed = next_env_state.info.get("is_crashed", jnp.zeros_like(next_env_state.done))
        
        r_nominal = next_env_state.info.get("r_joint_nominal", jnp.zeros_like(next_env_state.reward))
        decay_weight = jnp.where(global_iteration < 50, 1.0, 0.0)
        adjusted_reward = next_env_state.reward + (decay_weight * r_nominal)
        
        # 4. Store the Transition
        transition = Transition(
            obs=last_obs,
            action=raw_action,
            reward=adjusted_reward,
            done=next_env_state.done,
            value=value,
            log_prob=log_prob,
            is_crashed=is_crashed
        )
        
        # 5. Pack state for the next loop iteration
        next_runner_state = (next_env_state, next_env_state.obs, key)
        return next_runner_state, transition

    # Initialize the loop and compile the scan
    initial_runner_state = (env_state, env_state.obs, rng)
    final_runner_state, trajectories = jax.lax.scan(
        _step, initial_runner_state, None, length=num_steps
    )
    
    return final_runner_state, trajectories