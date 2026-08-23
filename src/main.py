import jax
import jax.numpy as jnp
import time
import os
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

import os
import signal
import sys

def force_exit_handler(sig, frame):
    print("\n[!] Ctrl+C detected. Forcing clean exit...")
    os._exit(0)  # os._exit immediately terminates the process and releases GPU memory

signal.signal(signal.SIGINT, force_exit_handler)

def main():
    print("Initializing Unitree Go2 PPO Training Pipeline...")
    
    # 1. Setup Configuration 
    cfg = PPOConfig(
        num_envs=32,
        num_steps=128,
        num_epochs=3,
        num_minibatches=8,
        entropy_coef=0.05,
        lr_actor=3e-4,
        lr_critic=1e-3,
    )
    rng = jax.random.PRNGKey(42)
    rng, rng_env, rng_init = jax.random.split(rng, 3)

    # 2. Instantiate Environment & Wrappers
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    env = Go2Env(xml_path=xml_path)
    
    # --- UPGRADE: action_repeat=2 for 2x physical speed ---
    env = EpisodeWrapper(env, episode_length=1000, action_repeat=2) 
    env = AutoResetWrapper(env)
    env = VmapWrapper(env)      
    
    actor = Actor(action_dim=12)
    critic = Critic()

    # 3. Initialize State
    print("Compiling networks and resetting environment...")
    obs_dim = env.observation_size 
    actor_state, critic_state = create_train_states(actor, critic, obs_dim, cfg, rng_init)
    
    jit_reset = jax.jit(env.reset)
    env_state = jit_reset(jax.random.split(rng_env, cfg.num_envs))

    # --- UPGRADE: Checkpoint Resume Infrastructure ---
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

    total_iterations = 1000
    print("Starting Training Loop!")
    print("-" * 80)
    
    global_start_time = time.time()
    
    for i in range(start_iteration, total_iterations):
        start_time = time.time()
        
        rng, iter_rng = jax.random.split(rng)
        
        # Execute the compiled monolithic training step
        # Note: The first iteration will pause to compile the graph.
        env_state, actor_state, critic_state, metrics = compiled_train_step(
            env, env_state, actor_state, critic_state, actor, critic, cfg, iter_rng
        )
        
        step_time = time.time() - start_time
        total_elapsed = time.time() - global_start_time
        # 4. Metric Tracking
        p_loss = jnp.mean(metrics["policy_loss"])
        v_loss = jnp.mean(metrics["value_loss"])
        entropy = jnp.mean(metrics["entropy"])
        mean_reward = jnp.mean(metrics["mean_reward"])
        crashes = int(jnp.sum(metrics["total_crashes"]))
        
        # Extract mean reward directly from the environment state metrics!
        # AutoResetWrapper accumulates episode returns in env_state.metrics
        
        # FPS = (num_envs * num_steps) / time
        fps = (cfg.num_envs * cfg.num_steps) / step_time
        
        # Format total elapsed time as MM:SS
        mins, secs = divmod(int(total_elapsed), 60)
        
        print(f"Iter {i:04d} | Time: {mins:02d}:{secs:02d} | FPS: {fps:5.0f} | Rew: {mean_reward:6.2f} | Falls: {crashes:4d} | P_Loss: {p_loss: .3f} | V_Loss: {v_loss: .3f} | Ent: {entropy: .3f}")
        
        # Save every 50 iterations
        if i % 50 == 0 and i > 0:
            # Removed the 'args=' keyword wrapper
            checkpointer.save(os.path.join(ckpt_dir, f"step_{i}"), actor_state.params)
            print(f"--> Saved checkpoint at iteration {i}")

if __name__ == "__main__":
    main()