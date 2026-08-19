import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

def main():
    print("Loading CPU MuJoCo model...")
    model_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    
    # 1. Load the standard C-based CPU model
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    
    # --- THE FIX: Dynamic Geometry Conversion ---
    # MJX lacks Cylinder-Box collisions. We iterate through the C-array 
    # and cast all Cylinders (7) into Capsules (6) before pushing to JAX.
    for i in range(mj_model.ngeom):
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mj_model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE
            
    mj_data = mujoco.MjData(mj_model)

    # 2. Transfer the model and data structures to the GPU via MJX
    print("Transferring memory to GPU via MJX...")
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # 3. JIT-compile the physics step function
    @jax.jit
    def step_fn(model, data, action):
        data = data.replace(ctrl=action)
        return mjx.step(model, data)

    # 4. Execute a single JAX-accelerated step
    dummy_action = jnp.zeros(mjx_model.nu)
    
    print("JIT-compiling and executing step (this takes a moment on the first run)...")
    new_data = step_fn(mjx_model, mjx_data, dummy_action)

    print("\n--- MJX System Diagnostics ---")
    print(f"Compute Devices Available: {jax.devices()}")
    print(f"Action Dimension (nu): {mjx_model.nu} (12 leg motors)")
    print(f"State Dimension (nq):  {mjx_model.nq}")
    print("Status: SUCCESS. MJX GPU physics step executed seamlessly.")

if __name__ == "__main__":
    main()