### physical parameters of the robot, for simulation (in RL it would be The robot)


from communicator import Communicator

from .object import Object
from .real_actuator import WheelsActuator
from .real_sensor import FinnishSensor, LidarSensor, CameraSensor, RealSensorArray

class RealRobot(Object):
    def __init__(self, bus: Communicator):
        super().__init__()  
        self.anchored = False
        self.model_name = "epuck2"
        self.initial_position = [- 0.2, - 0.375, 0.05]
        self.initial_orientation = [0, 0, 0, 1] 
        self.bus = bus
        self.sensors = {
            "lidars": RealSensorArray(self.bus, [LidarSensor(self.bus, ang, 0.1) for ang in [17, 50, 90, 150, 210, 270, 310, 343]]),
            "camera": CameraSensor(self.bus, [120, 160, 3]),
            "coords": FinnishSensor(self.bus)
            }
        self.actuators = {
            "wheels": WheelsActuator(self.bus)
        }
    def set_end_pad_coords(self, finnish_point):
        self.sensors["coords"].end_pad_coords = finnish_point

    def setID(self, id):
        super().setID(id)
        for s in self.sensors.values():
            s.set_robot_id(id)
        for a in self.actuators.values():
            a.set_robot_id(id)
        