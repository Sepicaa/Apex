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
        
        # Master sitting posture reference
        self.q_master_sit = jnp.array([
             0.00,  1.10, -1.45, 
             0.00,  1.10, -1.45,
             0.00, -0.65, -2.70,
             0.00, -0.65, -2.70
        ])
        
        # 100 perturbed sitting references (+/- 0.05 rad)
        rng_dataset = jax.random.PRNGKey(0)
        self.sit_dataset = self.q_master_sit + jax.random.uniform(
            rng_dataset, (100, 12), minval=-0.05, maxval=0.05
        )

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
            commands  # 4D command: [v_x, v_y, omega_z, is_sitting] (Total: 49D)
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
        
        # Command Sampling
        is_sitting = jax.random.bernoulli(rng_sit, p=0.2).astype(jnp.float32)
        
        angle = jax.random.uniform(rng_cmd, minval=-jnp.pi, maxval=jnp.pi)
        speed = jax.random.uniform(rng_cmd, minval=0.0, maxval=1.0)
        v_x = jnp.cos(angle) * speed
        v_y = jnp.sin(angle) * speed
        omega_z = jax.random.uniform(rng_cmd, minval=-0.5, maxval=0.5)

        initial_commands = jnp.array([v_x, v_y, omega_z, is_sitting])
        target_sit_pose = jax.random.choice(rng_sit, self.sit_dataset)
        
        obs = self._get_obs(pipeline_state, initial_action, initial_commands)
        
        info = {
            "last_action": initial_action,
            "commands": initial_commands,
            "target_sit_pose": target_sit_pose,
            "step_count": jnp.array(0),
            "is_crashed": jnp.array(0.0),
            "progress": jnp.array(0.0)
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
        # --- 1. Action Scaling (Widened Range) ---
        # Allow +/- 0.8 rad (~45.8 deg) deviation so sitting/turning poses are reachable
        action_scaled = action * 0.8
        target_angles = self.q_nom + action_scaled
        
        progress = state.info["progress"]
        
        # --- 2. Physics Simulation ---
        pipeline_state = self.pipeline_step(state.pipeline_state, target_angles)
        data = pipeline_state
        
        # --- 3. Extract Kinematics ---
        v_base = data.qvel[:3]
        omega_base = data.qvel[3:6]
        z_height = data.qpos[2]
        q_joints = data.qpos[7:19]
        
        quat = data.qpos[3:7]
        g_proj = math.rotate(jnp.array([0.0, 0.0, -1.0]), math.quat_inv(quat))
        
        commands = state.info["commands"]
        last_action = state.info["last_action"]
        target_sit_pose = state.info["target_sit_pose"]
        
        # --- 4. Termination Condition ---
        is_flipped = g_proj[2] > 0.0
        is_bottomed_out = z_height < 0.20
        is_crashed_bool = jnp.logical_or(is_flipped, is_bottomed_out)
        
        done = jnp.where(is_crashed_bool, 1.0, 0.0)
        
        # --- 5. Reward Calculation ---
        reward = self._calc_reward(
            v_base, omega_base, action, last_action, 
            commands, g_proj, z_height, q_joints, target_sit_pose, is_crashed_bool, progress
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
                    z_height: jax.Array, q_joints: jax.Array, 
                    target_sit_pose: jax.Array, has_crashed: jax.Array,
                    progress: jax.Array) -> jax.Array:
                    
        # 1. Alive Bonus (Always active)
        is_upright = (g_proj[2] < -0.85) & (z_height > 0.22)
        r_alive = jnp.where(is_upright & ~has_crashed, 0.5, 0.0)

        # 2. Stage 1: Standing Objective (Active early, fades out)
        # Penalize moving away from the default standing joint angles
        stand_error = jnp.sum(jnp.square(q_joints - self.q_nom))
        r_stand = jnp.exp(-stand_error / 0.1) * 2.0
        
        # 3. Stage 2: Walking Objective (Fades in after 20% progress)
        linear_error = jnp.sum(jnp.square(v_base[:2] - commands[:2]))
        angular_error = jnp.square(omega_base[2] - commands[2])
        r_walk = 2.0 * jnp.exp(-linear_error / 0.25) + 1.0 * jnp.exp(-angular_error / 0.25)
        
        # 4. Stage 3: Good Form Regularization (Fades in after 70% progress)
        r_smooth = -jnp.sum(jnp.square(action - last_action)) * 0.05
        
        # --- The Continuous JAX Curriculum Logic ---
        
        # If progress < 0.2, rely heavily on r_stand. Otherwise, rely on r_walk.
        task_reward = jnp.where(progress < 0.2, r_stand, r_walk)
        
        # Only apply the smoothness penalty in the final 30% of training
        style_penalty = jnp.where(progress > 0.7, r_smooth, 0.0)
        
        return r_alive + task_reward + style_penalty