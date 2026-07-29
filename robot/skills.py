# all basic skills
import time
import math

from .virtual_actuator import *
from .virtual_sensor import *

class Skill():
    '''
    parent class for skills, each skill have set of sensors/actuators it requires
    '''
    def __init__(self):
        pass

    def execute():
        ''' executes the skill from start to finish and returns if the execution was completed '''
        raise NotImplementedError("The execute() method must be implemented in the subclass.") 



### --- EXTRA CALCULATIONS FOR EPUCK2   --- ###
T_90_CONST = math.pi*0.052/(4*0.02) 
### ---                                 --- ###
class MoveForward(Skill):
    ''' moves forward until it aproaches a wall '''
    def __init__(self, lidar_forward_left: LidarSensor, lidar_forward_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float):
        '''
        lidar_forward_left - a lidar with angle (0-30) deg
        lidar_forward_right - a lidar with angle (330-360) deg
        wheels - a set of wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity 
        '''
        super().__init__()
        self.lidar_fl = lidar_forward_left
        self.lidar_fr = lidar_forward_right
        self.wheels = wheels

        self.velocity = velocity
        self.min_dist = min_dist

    def execute(self):
        self.wheels.set_parameters(self.velocity, self.velocity)
        while True:
            if self.lidar_fl.read() < self.min_dist or self.lidar_fr.read() < self.min_dist: 
                return True

class TurnLeft(Skill):
    '''
    left turn 90 deg
    '''
    def __init__(self, lidar_forward_left: LidarSensor, lidar_forward_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float):
        '''
        lidar_forward_left - a lidar with angle (0-30) deg
        lidar_forward_right - a lidar with angle (330-360) deg
        wheels - a set of 2 wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity 
        '''
        super().__init__()
        self.lidar_fl = lidar_forward_left
        self.lidar_fr = lidar_forward_right
        self.wheels = wheels

        self.velocity = velocity
        self.min_dist = min_dist
        self.turn_duration =T_90_CONST/velocity # calculations based on epuck2 (wheel radius and distance between them) 

    def execute(self):
        start_time = time.time()
        while time.time() - start_time < self.turn_duration:
            self.wheels.set_parameters(-self.velocity, self.velocity)
            time.sleep(0.01)  # Prevent CPU spinning

        self.wheels.set_parameters(0.0, 0.0) # halt
        return True

class TurnRight(Skill):
    '''
    right turn 90 deg
    '''
    def __init__(self, lidar_forward_left: LidarSensor, lidar_forward_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float):
        '''
        lidar_forward_left - a lidar with angle (0-30) deg
        lidar_forward_right - a lidar with angle (330-360) deg
        wheels - a set of 2 wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity 
        '''
        super().__init__()
        self.lidar_fl = lidar_forward_left
        self.lidar_fr = lidar_forward_right
        self.wheels = wheels

        self.velocity = velocity
        self.min_dist = min_dist
        self.turn_duration = T_90_CONST/velocity # calculations based on epuck2 (wheel radius and distance between them) 

    def execute(self):
        start_time = time.time()
        while time.time() - start_time < self.turn_duration:
            self.wheels.set_parameters(self.velocity, -self.velocity)
            time.sleep(0.01)  # Prevent CPU spinning

        self.wheels.set_parameters(0.0, 0.0) # halt
        return True


