import time
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer

from src.envs.go2_env import Go2Env

def main():
    print("Initializing Go2 Environment (JIT Compiled)...")
    
    # 1. Initialize the JAX Environment
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    env = Go2Env(xml_path=xml_path)
    
    # 2. Setup standard CPU MuJoCo for the Viewer
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    for i in range(mj_model.ngeom):
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mj_model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    mj_data = mujoco.MjData(mj_model)

    # 3. JIT Compile Reset and Step
    print("Compiling JAX step functions (this will take a few seconds)...")
    rng = jax.random.PRNGKey(42)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    state = jit_reset(rng)
    zero_action = jnp.zeros(12)
    
    print("Launching MuJoCo Viewer! Close the window to exit.")
    print("-" * 60)
    
    # 4. Render Loop
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Step the JAX physics engine (blazing fast)
            state = jit_step(state, zero_action)
            
            # Sync the calculated JAX states back to the CPU for rendering
            mj_data.qpos[:] = np.asarray(state.pipeline_state.qpos)
            mj_data.qvel[:] = np.asarray(state.pipeline_state.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            
            viewer.sync()
            
            # --- EXTRACT METRICS ---
            # qpos[0:3] contains the global X, Y, Z coordinates of the base
            x, y, z = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
            
            # Print telemetry inline (overwrites the same line to prevent spam)
            print(f"\rBase Position -> X: {x: 6.3f} | Y: {y: 6.3f} | Z (Height): {z: 6.3f} ", end="")
            
            # Throttle the loop to run at real-time 1x speed (0.02s per step)
            time_until_next_step = 0.02 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
            # If the robot crashes, reset it
            # Convert JAX array to Python boolean to evaluate
            if bool(np.asarray(state.done)):
                print("\n[!] Robot crashed! Resetting...")
                rng, sub_rng = jax.random.split(rng)
                state = jit_reset(sub_rng)
                time.sleep(0.5)

if __name__ == "__main__":
    main()