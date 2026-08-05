### Simulated arena

from .object import Object
import pybullet as p

class Arena(Object):
    def __init__(self):
        super().__init__()  
        self.model_name = "arena"
        self.initial_position = [0, 0, 0]
        self.initial_orientation = [0, 0, 0, 1] 
        self.end_pad_coords = None

    def sense(self):
        print("EnvArena: sense() method called, but no sensors to sense.")
    def act(self):
        print("EnvArena: act() method called, but no actuators to act.")

    def get_end_pad_coords(self):
        joint_name_to_index = {}
    
        # Map all joint names to their indices
        for i in range(p.getNumJoints(self.id)):
            joint_info = p.getJointInfo(self.id, i)
            joint_name = joint_info[1].decode('utf-8') 
            joint_name_to_index[joint_name] = i
        
        # Get the index of the target joint
        end_pad_idx = joint_name_to_index.get("end_joint")
    
        # Safety check in case the joint doesn't exist in the URDF
        if end_pad_idx is None:
            print("Warning: 'end_joint' not found in URDF.")
            return None

        # Get link state (0 is world link position, 1 is world link orientation)
        link_state = p.getLinkState(self.id, end_pad_idx)
    
        # Extract the xyz coordinates
        self.end_pad_coords = link_state[0] 
    
        return self.end_pad_coords