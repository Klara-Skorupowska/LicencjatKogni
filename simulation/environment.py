### Environment class, ergo simulation, in RL unnecessary 
# it contains all pybullet stuff, like loading the arena, robot, etc.

import pybullet as p
import pybullet_data
import math

from .object import Object
from communicator import Communicator

class Environment():
    def __init__(self):
        self.objects = None

    def __init__(self, objects: list[Object] = []):
        self.objects = objects

    def add_object(self, obj: Object):
        self.objects.append(obj)

    def setup(self, com: Communicator, time_step: float):
        # Config physics and environment
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.resetDebugVisualizerCamera(1.0, 90, -120, [0,0,0])
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)

        # load all objects in the environment
        for obj in self.objects:
            urdf_path = obj.create_urdf_from_xacro(obj.model_name)
            obj_id = p.loadURDF(urdf_path, obj.initial_position, obj.initial_orientation, useFixedBase=obj.anchored)
            obj.setID(obj_id)
            obj.set_communicator(com)
            obj.setup()
            obj.register_services()

        self._draw_door_zone()
        self._draw_goal_zone()

    def _draw_door_zone(self):
        # Z-height slightly above the floor (0.02) so it doesn't clip into the ground
        z = 0.02 
        color = [0.40, 0.80, 0.40]  # Like a door
        width = 2.0        # Line thickness

        # --- Define the Points ---
        # Left room boundary (depth 0.5, width 0.4)
        left_top = [-0.25, 0.2, z]
        left_bot = [-0.25, -0.2, z]
    
        # Doorway boundary (center 0, width 0.2)
        door_top = [0.0, 0.1, z]
        door_bot = [0.0, -0.1, z]
    
        # Right room boundary (depth 0.5, width 0.4)
        right_top = [0.25, 0.2, z]
        right_bot = [0.25, -0.2, z]

        # --- Draw the Lines ---
        # Left Room Trapezoid
        p.addUserDebugLine(left_top, door_top, color, width)  # Top diagonal
        p.addUserDebugLine(left_bot, door_bot, color, width)  # Bottom diagonal
        p.addUserDebugLine(left_top, left_bot, color, width)  # Far left edge
    
        # Right Room Trapezoid
        p.addUserDebugLine(door_top, right_top, color, width) # Top diagonal
        p.addUserDebugLine(door_bot, right_bot, color, width) # Bottom diagonal
        p.addUserDebugLine(right_top, right_bot, color, width)# Far right edge
    
        # Optional: Draw a green line right across the doorway threshold
        p.addUserDebugLine(door_top, door_bot, [0, 1, 0], width) 
    
    def _draw_goal_zone(self):
        # Center of the goal based on your URDF (floor_x*0.375, floor_y*0.45)
        cx = 0.375 
        cy = 0.45
    
        z = 0.02           # Slightly above the floor to prevent Z-fighting
        radius = 0.15      # Your max_dist threshold
        num_segments = 32  # Number of lines to make it look smooth
        color = [0.90, 0.80, 0.20]  # Like a goal
        width = 2.0        # Line thickness
    
        points = []
    
        # 1. Calculate all the points along the circumference
        for i in range(num_segments + 1):
            angle = (i / num_segments) * 2 * math.pi
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append([x, y, z])
        
        # 2. Draw lines connecting each point to the next
        for i in range(num_segments):
            p.addUserDebugLine(points[i], points[i+1], color, width)
    
    def update(self):
        pass

    def close(self):
        p.disconnect()

