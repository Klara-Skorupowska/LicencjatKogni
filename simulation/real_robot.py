### physical parameters of the robot, for simulation (in RL it would be The robot)

import pybullet as p

from communicator import Communicator

from .object import Object
from .real_actuator import WheelsActuator
from .real_sensor import LidarSensor, CameraSensor, RealSensorArray

class RealRobot(Object):
    def __init__(self, bus: Communicator):
        super().__init__()  
        self.anchored = False
        self.model_name = "epuck2"
        self.initial_position = [- 0.375, - 0.375, 0.05]
        self.initial_orientation = [0, 0, 0, 1] 
        self.bus = bus
        self.sensors = {
            "lidars": RealSensorArray(self.bus, [LidarSensor(self.bus, ang, 0.1) for ang in [17, 50, 90, 150, 210, 270, 310, 343]]),
            "camera": CameraSensor(self.bus, [120, 160, 3]),
            }
        self.actuators = {
            "wheels": WheelsActuator(self.bus)
        }

    def setID(self, id):
        super().setID(id)
        for s in self.sensors.values():
            s.set_robot_id(id)
        for a in self.actuators.values():
            a.set_robot_id(id)
    
    def register_services(self):
        self.bus.register_service(f"/realrobot/give_id", self.give_id)

    def setup(self):
        # camera setup
        width = self.sensors["camera"].width
        height = self.sensors["camera"].height
        fov = 60
        aspect = width / height
        near = 0.01 
        far = 10.0
        self.sensors["camera"].projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        name2idx =  self._joint_name_to_index()
        self.sensors["camera"].link_index =name2idx.get("camera_joint")
        pass
