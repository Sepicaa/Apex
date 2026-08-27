import jax
import jax.numpy as jnp
import mujoco
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from mujoco import mjx, mjtGeom

class Go2Env(PipelineEnv):
    def __init__(self, xml_path: str = "third_party/mujoco_menagerie/unitree_go2/scene_mjx.xml", **kwargs):
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] == mjtGeom.mjGEOM_CYLINDER:
                mj_model.geom_type[i] = mjtGeom.mjGEOM_CAPSULE
                
        mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        sys = mjcf.load_model(mj_model)
        
        self.q_nom = jnp.array([
            0.0, 0.9, -1.8,  # Front Left
            0.0, 0.9, -1.8,  # Front Right
            0.0, 0.9, -1.8,  # Rear Left
            0.0, 0.9, -1.8   # Rear Right
        ])
        
        self.action_scale = 0.5  
        self.target_height = 0.29 

        super().__init__(sys, backend='mjx', n_frames=5, **kwargs)

    def _get_obs(self, data: mjx.Data, action: jax.Array, commands: jax.Array) -> jax.Array:
        quat = data.qpos[3:7]
        inv_quat = math.quat_inv(quat)
        
        v_local = math.rotate(data.qvel[:3], inv_quat)
        omega_local = math.rotate(data.qvel[3:6], inv_quat)
        g_proj = math.rotate(jnp.array([0.0, 0.0, -1.0]), inv_quat)
        
        q_joints = data.qpos[7:19]
        dq_joints = data.qvel[6:18]
        
        obs = jnp.concatenate([
            v_local,                 
            omega_local,             
            g_proj,                  
            q_joints - self.q_nom,   
            dq_joints * 0.05,        
            action,                  
            commands                 
        ])
        obs = jnp.clip(obs, -10.0, 10.0)
        return jnp.nan_to_num(obs)                

    def reset(self, rng: jax.Array) -> State:
        # Added rng_phase to roll the dice for the curriculum stage
        rng, rng_noise, rng_speed, rng_angle, rng_yaw, rng_phase = jax.random.split(rng, 6)
        
        qpos = self.sys.qpos0
        qpos = qpos.at[2].set(self.target_height)
        qpos = qpos.at[7:19].set(
            self.q_nom + jax.random.uniform(rng_noise, (12,), minval=-0.05, maxval=0.05)
        )
        
        qvel = jnp.zeros(self.sys.nv)
        pipeline_state = self.pipeline_init(qpos, qvel)
        
        # --- OVERLAPPING CURRICULUM LOGIC ---
        # Roll a float between 0.0 and 1.0 to select the phase
        phase_sampler = jax.random.uniform(rng_phase, ())
        
        # Define the probability boundaries (e.g., 20% Stand, 30% Walk Forward, 50% Omni)
        is_phase_1 = phase_sampler < 0.20
        is_phase_2 = jnp.logical_and(phase_sampler >= 0.20, phase_sampler < 0.50)
        # Phase 3 covers the remaining 0.50 to 1.0
        is_phase_1 = 1
        
        # Generate the raw maximum bounds
        base_speed = jax.random.uniform(rng_speed, (), minval=0.0, maxval=1.2)
        omni_angle = jax.random.uniform(rng_angle, (), minval=-jnp.pi, maxval=jnp.pi) 
        raw_yaw = jax.random.uniform(rng_yaw, (), minval=-1.0, maxval=1.0)
        
        # Apply the phase masks to restrict the vectors
        # Phase 1: speed = 0, angle = 0, yaw = 0 (Standing Still)
        speed = jnp.where(is_phase_1, 0.0, base_speed)
        raw_yaw = jnp.where(is_phase_1, 0.0, raw_yaw)
        
        # Phase 2: angle is strictly 0.0 (Forward Walk Only). Phase 3 gets omni_angle.
        angle = jnp.where(is_phase_1, 0.0, jnp.where(is_phase_2, 0.0, omni_angle))
        
        # Calculate final velocity vectors
        v_x = speed * jnp.cos(angle)
        v_y = speed * jnp.sin(angle)
        
        speed_penalty = (speed / 1.2) * 0.85 
        omega_z = raw_yaw * (1.0 - speed_penalty)
        
        commands = jnp.array([v_x, v_y, omega_z])
        
        initial_action = jnp.zeros(12)
        obs = self._get_obs(pipeline_state, initial_action, commands)
        
        info = {
            "last_action": initial_action,
            "commands": commands,
            "step_count": jnp.array(0, dtype=jnp.int32),
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
        target_qpos = self.q_nom + action * self.action_scale
        pipeline_state = self.pipeline_step(state.pipeline_state, target_qpos)
        data = pipeline_state
        
        quat = data.qpos[3:7]
        inv_quat = math.quat_inv(quat)
        
        v_local = math.rotate(data.qvel[:3], inv_quat)
        omega_local = math.rotate(data.qvel[3:6], inv_quat)
        g_proj = math.rotate(jnp.array([0.0, 0.0, -1.0]), inv_quat)
        
        base_z = data.qpos[2]
        commands = state.info["commands"]
        last_action = state.info["last_action"]
        
        has_nans = jnp.any(jnp.isnan(data.qvel)) | jnp.any(jnp.isnan(data.qpos))
        
        foot_forces = data.sensordata[-9:-5]
        num_feet_touching = jnp.sum(foot_forces > 0.1)
        has_illegal_touch = jnp.any(data.sensordata[-5:] > 0.1)
        
        is_inverted = g_proj[2] > -0.4 
        
        is_crashed = jnp.logical_or(is_inverted, has_illegal_touch)
        is_crashed = jnp.logical_or(is_crashed, has_nans)
        
        done = jnp.where(is_crashed, 1.0, 0.0)
        
        reward = self._calc_reward(
            v_local, omega_local, g_proj, base_z,
            data.qpos[7:19], data.qvel[6:18],
            action, last_action, commands, is_crashed, num_feet_touching
        )
        
        reward = jnp.where(has_nans, -1.0, reward)
        
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

    def _calc_reward(
        self, v_local, omega_local, g_proj, base_z,
        q_joints, dq_joints, action, last_action,
        commands, is_crashed, num_feet_touching
    ) -> jax.Array:
        
        lin_vel_error = jnp.sum(jnp.square(v_local[:2] - commands[:2]))
        r_lin_vel = jnp.exp(-lin_vel_error / 0.25) * 3.5
        
        ang_vel_error = jnp.square(omega_local[2] - commands[2])
        r_ang_vel = jnp.exp(-ang_vel_error / 0.25) * 1.8
        
        r_z_vel = -jnp.square(v_local[2]) * 2.0
        r_ang_rates = -jnp.sum(jnp.square(omega_local[:2])) * 0.05
        r_flat_posture = -jnp.sum(jnp.square(g_proj[:2])) * 2.5
        r_height = -jnp.square(base_z - self.target_height) * 10.0
        r_action_rate = -jnp.sum(jnp.square(action - last_action)) * 0.02
        r_joint_vel = -jnp.sum(jnp.square(dq_joints)) * 0.0001
        r_joint_nominal = -jnp.sum(jnp.square(q_joints - self.q_nom)) * 0.01
        
        r_airborne = jnp.where(num_feet_touching == 0, -0.2, 0.0)
        
        r_alive = jnp.where(is_crashed, -1.0, 0.5)
        
        total_reward = (
            r_lin_vel +
            r_ang_vel +
            r_z_vel +
            r_ang_rates +
            r_flat_posture +
            r_height +
            r_action_rate +
            r_joint_vel +
            r_joint_nominal +
            r_airborne +
            r_alive
        )
        
        return jnp.clip(total_reward, -40.0, 70.0) * 0.02