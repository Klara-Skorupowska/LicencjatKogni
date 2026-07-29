### physical parameters of the robot, for simulation (in RL it would be The robot)


from communicator import Communicator

from .object import Object
from .real_actuator import WheelsActuator
from .real_sensor import LidarSensor, CameraSensor

class RealRobot(Object):
    def __init__(self, bus: Communicator):
        super().__init__()  
        self.anchored = False
        self.model_name = "epuck2"
        self.initial_position = [-0.375, 0.3, 0.05]
        self.initial_orientation = [0, 0, 0, 1] 
        self.bus = bus
        self.lasers = [LidarSensor(self.bus, ang, 0.1) for ang in [17, 50, 90, 150, 210, 270, 310, 343]]
        self.wheels = WheelsActuator(self.bus)
        self.camera = CameraSensor(self.bus, [240, 320, 3])

    def setID(self, id):
        super().setID(id)
        self.wheels.set_id(id)
        self.camera.robot_id = id
        for laser in self.lasers:
            laser.robot_id = id
        