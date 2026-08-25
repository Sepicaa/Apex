import jax
import jax.numpy as jnp
import mujoco
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from mujoco import mjx, mjtGeom

class Go2Env(PipelineEnv):
    def __init__(self, xml_path: str, terrain_mode: str = "flat", **kwargs):
        # 1. Load raw MuJoCo C-model
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
        # 2. Patch CYLINDER to CAPSULE to prevent MJX compilation errors
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] == mjtGeom.mjGEOM_CYLINDER:
                mj_model.geom_type[i] = mjtGeom.mjGEOM_CAPSULE
                
        # 3. Patch friction cone to Pyramidal (MJX compatibility)
        mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
                
        # 4. Pass patched model to Brax
        sys = mjcf.load_model(mj_model)
        
        self.terrain_mode = terrain_mode
        
        # Nominal standing posture (rad)
        self.q_nom = jnp.array([
             0.1,  0.8, -1.5,  # Front Right
            -0.1,  0.8, -1.5,  # Front Left
             0.1,  1.0, -1.5,  # Rear Right
            -0.1,  1.0, -1.5   # Rear Left
        ])
        
        # Removed the sit_dataset and q_master_sit arrays since 
        # the task is now purely continuous locomotion.

        super().__init__(sys, backend='mjx', n_frames=10, **kwargs)

    def _get_obs(self, data: mjx.Data, action: jax.Array, commands: jax.Array) -> jax.Array:
        v_base = data.qvel[:3]
        omega_base = data.qvel[3:6]
        
        quat = data.qpos[3:7]
        g_proj = math.rotate(jnp.array([0.0, 0.0, -1.0]), math.quat_inv(quat))
        
        q_joints = data.qpos[7:19]
        dq_joints = data.qvel[6:18]
        
        return jnp.concatenate([
            v_base,
            omega_base,
            g_proj,
            q_joints - self.q_nom,
            dq_joints,
            action,
            commands  # 3D command: [v_x, v_y, omega_z] (Total: 48D)
        ])

    def reset(self, rng: jax.Array) -> State:
        rng, rng_pos, rng_vel, rng_cmd, rng_sit = jax.random.split(rng, 5)
        
        # Add slight initial posture perturbation
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng_pos, (self.sys.nq,), minval=-0.01, maxval=0.01
        )
        qpos = qpos.at[7:19].set(self.q_nom + jax.random.uniform(
            rng_pos, (12,), minval=-0.05, maxval=0.05
        ))
        
        qvel = jax.random.uniform(
            rng_vel, (self.sys.nv,), minval=-0.01, maxval=0.01
        )
        
        pipeline_state = self.pipeline_init(qpos, qvel)
        initial_action = jnp.zeros(12)
        
        # Command Sampling (Removed 'is_sitting', focusing purely on locomotion)
        angle = jax.random.uniform(rng_cmd, minval=-jnp.pi, maxval=jnp.pi)
        speed = jax.random.uniform(rng_cmd, minval=0.0, maxval=1.0)
        v_x = jnp.cos(angle) * speed
        v_y = jnp.sin(angle) * speed
        omega_z = jax.random.uniform(rng_cmd, minval=-0.5, maxval=0.5)

        initial_commands = jnp.array([v_x, v_y, omega_z])
        
        obs = self._get_obs(pipeline_state, initial_action, initial_commands)
        
        # Removed the buggy "progress" tracking
        info = {
            "last_action": initial_action,
            "commands": initial_commands,
            "step_count": jnp.array(0),
            "is_crashed": jnp.array(0.0),
        }
        
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            metrics={},
            info=info
        )

    def step(self, state: State, action: jax.Array) -> State:
        # --- 1. Action Scaling (The Sweet Spot) ---
        # Reverted to 0.5 to prevent violent twitches per Hugging Face spec
        action_scaled = action * 0.5
        target_angles = self.q_nom + action_scaled
        
        # --- 2. Physics Simulation ---
        pipeline_state = self.pipeline_step(state.pipeline_state, target_angles)
        data = pipeline_state
        
        # --- 3. Extract Kinematics ---
        v_base = data.qvel[:3]
        omega_base = data.qvel[3:6]
        z_height = data.qpos[2]
        
        quat = data.qpos[3:7]
        g_proj = math.rotate(jnp.array([0.0, 0.0, -1.0]), math.quat_inv(quat))
        
        commands = state.info["commands"]
        last_action = state.info["last_action"]
        
        # --- 4. Termination Condition ---
        is_flipped = g_proj[2] > 0.0
        is_bottomed_out = z_height < 0.20
        is_crashed_bool = jnp.logical_or(is_flipped, is_bottomed_out)
        
        done = jnp.where(is_crashed_bool, 1.0, 0.0)
        
        # --- 5. Reward Calculation ---
        reward = self._calc_reward(
            v_base, omega_base, action, last_action, 
            commands, g_proj, is_crashed_bool
        )
        
        # --- 6. Next Observation & Memory Update ---
        obs = self._get_obs(pipeline_state, action, commands)
        
        info = state.info
        info["last_action"] = action
        info["step_count"] += 1
        info["is_crashed"] = done
        
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=info
        )

    def _calc_reward(self, v_base: jax.Array, omega_base: jax.Array, 
                     action: jax.Array, last_action: jax.Array, 
                     commands: jax.Array, g_proj: jax.Array, 
                     has_crashed: jax.Array) -> jax.Array:
                     
        # 1. Positive Tracking Rewards (Exp scale so max is 1.0)
        linear_error = jnp.sum(jnp.square(v_base[:2] - commands[:2]))
        angular_error = jnp.square(omega_base[2] - commands[2])
        
        r_lin_track = jnp.exp(-linear_error / 0.25) * 1.0
        r_ang_track = jnp.exp(-angular_error / 0.25) * 1.0
        
        # 2. Huge Crash Penalty
        r_crash = jnp.where(has_crashed, -200.0, 0.0)
        
        # 3. Flat Orientation Penalty (Penalize tilt away from z-axis)
        r_flat = -jnp.sum(jnp.square(g_proj[:2])) * 5.0
        
        # 4. Action Rate L2 (Smoothness penalty)
        r_action_rate = -jnp.sum(jnp.square(action - last_action)) * 0.05
        
        # Sum everything
        total_reward = r_lin_track + r_ang_track + r_crash + r_flat + r_action_rate
        
        # --- THE MAGIC BULLET: SCALE BY DT ---
        # Assuming a step_dt of roughly 0.02 seconds
        dt = 0.02
        return total_reward * dt