import mujoco
import mujoco.viewer
import numpy as np

def main():
    print("--- Loading MuJoCo Model ---")
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    print("--- Launching Interactive Viewer ---")
    print("INSTRUCTIONS:")
    print("1. Press 'Spacebar' immediately to pause physics (so it doesn't fall).")
    print("2. On the right-hand menu, expand 'Physics' and disable 'Gravity'.")
    print("3. Expand the 'Joints' tab on the right.")
    print("4. Use the sliders to manually bend the robot into a sitting pose.")
    print("5. Close the window when finished.")
    
    # This blocks execution until you close the GUI window
    mujoco.viewer.launch(model, data)
    
    print("\n--- Extracted q_master Array ---")
    # qpos[7:19] isolates the 12 leg motors from the base coordinates
    q_master = np.array(data.qpos[7:19])
    
    print("Copy this array into your JAX environment:")
    print(np.repr(q_master))

if __name__ == "__main__":
    main()