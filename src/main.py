import jax
import jax.numpy as jnp
import time
import os
import signal
import sys
import numpy as np
from collections import deque
from functools import partial
import orbax.checkpoint as ocp

from brax.envs.wrappers.training import VmapWrapper, AutoResetWrapper, EpisodeWrapper

from src.envs.go2_env import Go2Env
from src.training.networks import Actor, Critic
from src.training.train_ppo import PPOConfig, create_train_states, ppo_train_iteration

@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def compiled_train_step(env, env_state, actor_state, critic_state, actor, critic, cfg, rng):
    return ppo_train_iteration(
        env, env_state, actor_state, critic_state, actor, critic, cfg, rng
    )

def force_exit_handler(sig, frame):
    print("\n[!] Ctrl+C detected. Forcing clean exit...")
    os._exit(0)

signal.signal(signal.SIGINT, force_exit_handler)

def main():
    print("Initializing Unitree Go2 PPO Training Pipeline...")
    
    cfg = PPOConfig(
        num_envs=64,
        num_steps=128,
        num_epochs=5,
        num_minibatches=8,
        entropy_coef=0.002,
        lr_actor=5e-4,
        lr_critic=5e-4,
    )
    rng = jax.random.PRNGKey(42)
    rng, rng_env, rng_init = jax.random.split(rng, 3)

    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene_mjx.xml"
    env = Go2Env(xml_path=xml_path)
    
    env = EpisodeWrapper(env, episode_length=1000, action_repeat=2) 
    env = AutoResetWrapper(env)
    env = VmapWrapper(env)      
    
    actor = Actor(action_dim=12)
    critic = Critic()

    print("Compiling networks and resetting environment...")
    obs_dim = env.observation_size 
    actor_state, critic_state = create_train_states(actor, critic, obs_dim, cfg, rng_init)
    
    jit_reset = jax.jit(env.reset)
    env_state = jit_reset(jax.random.split(rng_env, cfg.num_envs))

    ckpt_dir = os.path.abspath("./checkpoints")
    checkpointer = ocp.StandardCheckpointer()
    start_iteration = 0
    
    if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
        existing_ckpts = [int(d.split('_')[1]) for d in os.listdir(ckpt_dir) if d.startswith('step_')]
        if existing_ckpts:
            latest_step = max(existing_ckpts)
            print(f"[*] Found existing checkpoint! Resuming from Iteration {latest_step}...")
            
            restored_params = checkpointer.restore(os.path.join(ckpt_dir, f"step_{latest_step}"))
            actor_state = actor_state.replace(params=restored_params)
            start_iteration = latest_step + 1

    total_iterations = 2500
    print("Starting Training Loop!")
    print("-" * 80)
    
    # Initialize trackers for the last 100 episodes
    ep_return_queue = deque(maxlen=100)
    ep_crash_queue = deque(maxlen=100)
    
    # Running tallies for the 64 parallel environments
    running_returns = np.zeros(cfg.num_envs)
    running_steps = np.zeros(cfg.num_envs)

    global_start_time = time.time()
    
    for i in range(start_iteration, total_iterations):
        start_time = time.time()
        
        rng, iter_rng = jax.random.split(rng)
        
        env_state, actor_state, critic_state, metrics = compiled_train_step(
            env, env_state, actor_state, critic_state, actor, critic, cfg, iter_rng
        )
        
        step_time = time.time() - start_time
        total_elapsed = time.time() - global_start_time
        
        # 1. Convert JAX arrays to NumPy for Python-side iteration
        step_rewards = np.array(metrics["step_rewards"])
        step_dones = np.array(metrics["step_dones"])
        
        # 2. Accumulate metrics step-by-step
        for t in range(cfg.num_steps):
            running_returns += step_rewards[t]
            running_steps += 1
            
            dones_at_t = step_dones[t]
            if np.any(dones_at_t):
                # Identify which specific environments finished
                finished_envs = dones_at_t > 0.5
                
                # A fall is any episode that terminates before the 1000-step timeout
                falls = (running_steps[finished_envs] < 1000).astype(float)
                
                # Push the completed episode totals to our rolling queues
                ep_return_queue.extend(running_returns[finished_envs].tolist())
                ep_crash_queue.extend(falls.tolist())
                
                # Reset the running tally for those specific environments
                running_returns[finished_envs] = 0.0
                running_steps[finished_envs] = 0.0
                
        # 3. Calculate rolling averages
        mean_reward = np.mean(ep_return_queue) if len(ep_return_queue) > 0 else 0.0
        mean_crashes = np.mean(ep_crash_queue) if len(ep_crash_queue) > 0 else 0.0
        
        p_loss = jnp.mean(metrics["policy_loss"])
        v_loss = jnp.mean(metrics["value_loss"])
        entropy = jnp.mean(metrics["entropy"])
        
        fps = (cfg.num_envs * cfg.num_steps) / step_time
        mins, secs = divmod(int(total_elapsed), 60)
        
        print(f"Iter {i:04d} | Time: {mins:02d}:{secs:02d} | FPS: {fps:5.0f} | Rew: {mean_reward:6.2f} | Falls/Ep: {mean_crashes:4.2f} | P_Loss: {p_loss: .3f} | V_Loss: {v_loss: .3f} | Ent: {entropy: .3f}")
        
        if i % 200 == 0 and i > 0:
            checkpointer.save(os.path.join(ckpt_dir, f"step_{i}"), actor_state.params)
            print(f"--> Saved checkpoint at iteration {i}")

if __name__ == "__main__":
    main()