import jax
import jax.numpy as jnp
import mujoco
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from mujoco import mjx, mjtGeom

class Go2Env(PipelineEnv):
    def __init__(self, xml_path: str, terrain_mode: str = "flat", **kwargs):
        # 1. Load the raw MuJoCo C-model
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
        # 2. Patch CYLINDER to CAPSULE to prevent MJX compilation errors
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] == mjtGeom.mjGEOM_CYLINDER:
                mj_model.geom_type[i] = mjtGeom.mjGEOM_CAPSULE
                
        # 3. Patch the friction cone to Pyramidal (MJX does not support Elliptic)
        mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
                
        # 4. Pass the patched model to Brax
        sys = mjcf.load_model(mj_model)
        
        self.terrain_mode = terrain_mode
        
        # Nominal standing posture (rads)
        self.q_nom = jnp.array([
             0.1,  0.8, -1.5,  # Front Right
            -0.1,  0.8, -1.5,  # Front Left
             0.1,  1.0, -1.5,  # Rear Right
            -0.1,  1.0, -1.5   # Rear Left
        ])
        
        # --- NEW: Imitation Learning Dataset ---
        self.q_master_sit = jnp.array([
             0.00,  1.10, -1.45, 
             0.00,  1.10, -1.45,
             0.00, -0.65, -2.70,
             0.00, -0.65, -2.70
        ])
        
        # Generate 100 poses with +/- 0.05 rad noise
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
            commands  # <-- MUST BE ADDED! (Bumps observation space to 49 dims)
        ])

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        # 1. Split the random number generator into 4 parts now
        rng, rng_pos, rng_vel, rng_sit = jax.random.split(rng, 4)
        
        # 2. Initialize positions and velocities with slight noise
        # qpos has 19 dims: [3 for base pos, 4 for base quat, 12 for joints]
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng_pos, (self.sys.nq,), minval=-0.01, maxval=0.01
        )
        
        # Override the joint positions to match our specific nominal posture
        qpos = qpos.at[7:19].set(self.q_nom + jax.random.uniform(
            rng_pos, (12,), minval=-0.05, maxval=0.05
        ))
        
        # DEBUG
        # qpos = qpos.at[2].set(10.0)
        
        # qvel has 18 dims: [3 linear, 3 angular, 12 joint velocities]
        qvel = jax.random.uniform(
            rng_vel, (self.sys.nv,), minval=-0.01, maxval=0.01
        )
        
        # 3. Initialize the physics pipeline
        pipeline_state = self.pipeline_init(qpos, qvel)
        
        # 4. Initialize our custom variables
        initial_action = jnp.zeros(12)
        # Commands are now 4D: [v_x, v_y, omega_z, is_sitting]
        
        # 1. 20% chance to Sit (1.0), 80% chance to Move (0.0)
        is_sitting = jax.random.bernoulli(rng_sit, p=0.2).astype(jnp.float32)

        # 2. Sample random angle and bounded magnitude for realistic 2D movement
        angle = jax.random.uniform(rng_vel, minval=-jnp.pi, maxval=jnp.pi)
        speed = jax.random.uniform(rng_vel, minval=0.0, maxval=1.0)
        v_x = jnp.cos(angle) * speed
        v_y = jnp.sin(angle) * speed
        
        # 3. Cap angular velocity
        omega_z = jax.random.uniform(rng_vel, minval=-0.5, maxval=0.5)

        initial_commands = jnp.array([v_x, v_y, omega_z, is_sitting])
        
        
        # Pick a random target sit pose from our dataset of 100
        target_sit_pose = jax.random.choice(rng_sit, self.sit_dataset)
        
        # 5. Generate the first observation
        obs = self._get_obs(pipeline_state, initial_action, initial_commands)
        
        # 6. Initialize reward and done flags
        reward = jnp.array(0.0)
        done = jnp.array(0.0)
        
        # 7. Store variables in the state info dictionary so they persist
        info = {
            "last_action": initial_action,
            "commands": initial_commands,
            "target_sit_pose": target_sit_pose, # Add this to memory!
            "step_count": jnp.array(0),
            "is_crashed": jnp.array(0.0)
        }
        
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics={},
            info=info
        )

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        
        # --- 1. Action Post-Processing (The Hardware Bridge) ---
        # Scale the normalized [-1, 1] action by 0.25 radians and add to nominal posture
        # action_scaled = action * 0.25
        action_scaled = action * 0.25
        target_angles = self.q_nom + action_scaled
        
        # --- 2. Physics Execution ---
        # Feed the target angles to MuJoCo's low-level PD controller
        pipeline_state = self.pipeline_step(state.pipeline_state, target_angles)
        data = pipeline_state
        
        # --- 3. Fetch Variables for Rewards ---
        v_base = data.qvel[:3]
        omega_base = data.qvel[3:6]
        z_height = data.qpos[2]
        q_joints = data.qpos[7:19]
        
        # Inner ear gravity projection
        quat = data.qpos[3:7]
        g_proj = math.rotate(jnp.array([0.0, 0.0, -1.0]), math.quat_inv(quat))
        
        commands = state.info["commands"]
        last_action = state.info["last_action"]
        target_sit_pose = state.info["target_sit_pose"]
        
        # --- 4. Calculate Rewards ---
        reward = self._calc_reward(
            v_base, omega_base, action, last_action, 
            commands, g_proj, z_height, q_joints, target_sit_pose
        )
        
        # --- 5. Advanced Done Condition ---
        # The exact same logic used in the reward function determines the episode end
        is_flipped = g_proj[2] > 0.0
        is_bottomed_out = z_height < 0.13
        # Define the boolean explicitly here
        is_crashed_bool = jnp.logical_or(is_flipped, is_bottomed_out)
        
        done = jnp.where(is_crashed_bool, 1.0, 0.0)
        
        # --- 6. Generate Next Observation ---
        obs = self._get_obs(pipeline_state, action, commands)
        
        # --- 7. Update State Memory ---
        info = state.info
        info["last_action"] = action
        info["step_count"] += 1
        info["is_crashed"] = jnp.where(is_crashed_bool, 1.0, 0.0).astype(jnp.float32)
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
                     target_sit_pose: jax.Array) -> jax.Array:
                     
        # --- 1. The Crash Penalty ---
        # Did it flip over (gravity pointing wrong way) or scrape its belly?
        is_flipped = g_proj[2] > 0.0
        is_bottomed_out = z_height < 0.13
        has_crashed = jnp.logical_or(is_flipped, is_bottomed_out)
        
        r_crash = jnp.where(has_crashed, -2.5, 0.0)

        # --- 2. The Sit vs. Move Branch ---
        # Assuming commands is now 4D: [v_x, v_y, omega_z, is_sitting]
        is_sitting_cmd = commands[3]
        
       # Branch A: Imitation Reward (Sitting)
        # 1. Joint Error: How close are the legs to the dataset pose?
        sit_joint_error = jnp.sum(jnp.square(q_joints - target_sit_pose))
        
        # 2. Height Error: The robot's torso must be resting near 0.22m
        sit_height_error = jnp.square(z_height - 0.22) * 50.0 
        
        # 3. Orientation Error: Ensure the robot is perfectly upright
        sit_upright_error = jnp.square(g_proj[2] - (-1.0)) * 10.0
        
        # 4. Station-Keeping Error: The robot must not slide or spin
        sit_vel_error = jnp.sum(jnp.square(v_base)) + jnp.sum(jnp.square(omega_base))
        
        # Combine errors. A high error in ANY category collapses the reward to 0.
        total_sit_error = sit_joint_error + sit_height_error + sit_upright_error + sit_vel_error
        r_sit = jnp.exp(-total_sit_error) * 2.0
        
        # Branch B: Tracking Reward (Moving)
        linear_error = jnp.sum(jnp.square(v_base[:2] - commands[:2]))
        angular_error = jnp.square(omega_base[2] - commands[2])
        r_tracking = jnp.exp(-linear_error) + jnp.exp(-angular_error)
        
        # Choose which reward to apply based on the boolean command
        r_task = jnp.where(is_sitting_cmd == 1.0, r_sit, r_tracking)
        
        # --- 3. The Universal Safety Penalty ---
        # Jitter is bad whether sitting or moving!
        r_smooth = -jnp.sum(jnp.square(action - last_action)) * 0.1
        
        return r_task + r_smooth + r_crash