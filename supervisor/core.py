import math
from communicator import Communicator
import numpy as np
import pybullet as p
import time

class Supervisor():
    '''
    supervisor is the omnipotent god: it tells the agent where it is and if the task were successful etc.
    it collects all the data
    It is yet another thread that uses communicator
    '''
    def __init__(self, bus: Communicator):
        self.bus = bus
        self.robot_id = self.bus.call_service(f"/realrobot/give_id")
        self.arena_id = self.bus.call_service(f"/arena/give_id")
        
        self.bus.register_service(f"/supervisor/ask/room_number", self.which_room)
        self.bus.register_service(f"/supervisor/ask/door_zone", self.door_zone)
        self.bus.register_service(f"/supervisor/do/door_zone", self.door_zone_teleport)
        self.bus.register_service(f"/supervisor/ask/goal_zone", self.goal_zone)

    def smth_callback(self, message):
        '''
        for when data appear in the ether self.bus.subscribe("/supervisor/attend/smth", self.smth_callback)
        '''
        pass 

    def smth_handler(self, request=None):
        '''
        for when agents ask a question (self.bus.register_service(f"/supervisor/ask/smth", self.smth_handler) or gives a task (self.bus.register_service(f"/supervisor/do/smth", self.smth_handler)
        '''
        return 0

    def which_room(self, request=None):
        '''
        for when agent ask a question: in which room am I.
        1: first room without the goal
        2: second room with the goal
        '''
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        point = np.array(position[:2])
        if point[0] < 0:
            return 1
        else:
            return 2

    def _joint_name_to_index(self, id):
        # Map all joint names to their indices
        joint_name_to_index = {}
        for i in range(p.getNumJoints(id)):
            joint_info = p.getJointInfo(id, i)
            joint_name = joint_info[1].decode('utf-8') 
            joint_name_to_index[joint_name] = i
        return joint_name_to_index

    def door_zone(self, request=None):
        # robot position
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        pos_robot = np.array(position[:2])

        # door position (hinge)
        pos_door = [0, -0.1]
        
        # Zone Configuration
        door_y_min = pos_door[1]           # -0.1
        door_y_max = pos_door[1] + 0.2     # 0.1
        door_x = pos_door[0]               # 0
        
        top_width = 0.2    # Width at the doorway
        bottom_width = 0.6 # Width at the far edge
        height = 0.5       # Depth the trapezoid extends into the room
        
        y = pos_robot[1]
        x = pos_robot[0]
        
        # 1. Calculate how far the robot is from the doorway edges (depth into the room)
        dy = 0
        if y > door_y_max:
            dy = y - door_y_max
        elif y < door_y_min:
            dy = door_y_min - y
            
        # 2. If the depth exceeds the trapezoid height, it's outside the zone
        if dy > height:
            return False
            
        # 3. Calculate the maximum allowed X width at this specific depth
        # It expands linearly from top_width to bottom_width as dy goes from 0 to height
        current_width = top_width + (bottom_width - top_width) * (dy / height)
        max_allowed_dx = current_width / 2.0
        
        # 4. Check if the robot's X is within this allowed width
        if abs(x - door_x) <= max_allowed_dx:
            return True
            
        return False

    def door_zone_teleport(self, request=None):

        # door position (hinge)
        pos_door = [0, -0.1]
        coeff = 1 if  self.which_room() == 2 else -1
        pos_robot = [
            pos_door[0] + (coeff * 0.1), 
            pos_door[1], 
            0.035
        ]
        dx = 0.0 - pos_robot[0]
        dy = 0.0 - pos_robot[1]
        yaw_angle = math.atan2(dy, dx)
        orn_robot = p.getQuaternionFromEuler([0, 0, yaw_angle])
        p.resetBasePositionAndOrientation(self.robot_id, pos_robot, orn_robot)
        time.sleep(1.0)

    def goal_zone(self, request=None):
        # robot position
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        pos_robot = np.array(position[:2])

        # goal position
        joint_name_to_index = self._joint_name_to_index(self.arena_id)
        goal_idx = joint_name_to_index.get("end_joint")
        if goal_idx is None:
            print("Warning: 'end_joint' not found in URDF.")
            return None
        link_state = p.getLinkState(self.arena_id, goal_idx)
        pos_door = link_state[0] 

        # the distance
        max_dist = 0.1
        x_dist = abs(pos_door[0] - pos_robot[0])
        y_dist = abs(pos_door[1] - pos_robot[1])
        dist = (x_dist**2+y_dist**2)**(1/2)

        return dist < max_dist
        