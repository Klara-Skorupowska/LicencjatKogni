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
        self.wheels.set_parameters([self.velocity, self.velocity])
        while True:
            if self.lidar_fl.read() < self.min_dist or self.lidar_fr.read() < self.min_dist: 
                self.wheels.set_parameters([0.0, 0.0]) # halt
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
            self.wheels.set_parameters([-self.velocity, self.velocity])
            time.sleep(0.01)  # Prevent CPU spinning

        self.wheels.set_parameters([0.0, 0.0]) # halt
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
            self.wheels.set_parameters([self.velocity, -self.velocity])
            time.sleep(0.01)  # Prevent CPU spinning

        self.wheels.set_parameters([0.0, 0.0]) # halt
        return True

class TurnAway(Skill):
    '''
    turn away from the obstacle ahead. Turning continues until the obstacle is at c. 120 deg (left or right) from the robot. 
    Skill fails if after the whole circle there is still something in front of the robot
    '''
    def __init__(self, lidar_front_left: LidarSensor, lidar_front_right: LidarSensor, lidar_back_left: LidarSensor, lidar_back_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float):
        '''
        lidar_front_left - a lidar with angle (0-30) deg
        lidar_front_right - a lidar with angle (330-360) deg
        lidar_back_left - a lidar with angle (90-120) deg
        lidar_back_right - a lidar with angle (240-270) deg
        wheels - a set of 2 wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity - the speed at which the robot should move
        '''
        super().__init__()
        self.lidar_front_left = lidar_front_left
        self.lidar_front_right = lidar_front_right
        self.lidar_back_left = lidar_back_left
        self.lidar_back_right = lidar_back_right
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist

    def execute(self):
        # calculate how long it takes to do a full turn (360 deg) based on the robot's parameters:
        full_circle_time = 4 * T_90_CONST / self.velocity 
        # the epsilon is acceptable error in distance to the obstacle 
        epsilon = 0.01
        # get the direction of the nearest obstacle (left or right) based on the front lidars:
        if self.lidar_front_left.read() < self.lidar_front_right.read():
            turn_direction = 'right'
            distance = self.lidar_front_left.read()
        else:
            turn_direction = 'left'
            distance = self.lidar_front_right.read()

        # turn until the distance from the front lidar appears on the back lidars (i.e. the obstacle is now behind us)
        start_time = time.time()
        while time.time() - start_time < full_circle_time:
            if turn_direction == 'right':
                self.wheels.set_parameters([self.velocity, -self.velocity])
                if self.lidar_back_left.read() < distance + epsilon or self.lidar_back_left.read() > distance - epsilon:
                    self.wheels.set_parameters([0.0, 0.0]) # halt
                    return True
            else:
                self.wheels.set_parameters([-self.velocity, self.velocity])
                if self.lidar_back_right.read() < distance + epsilon or self.lidar_back_right.read() > distance - epsilon: # the obstacle is now behind us
                    self.wheels.set_parameters([0.0, 0.0]) # halt
                    return True
                
            time.sleep(0.01)  # Prevent CPU spinning
        self.wheels.set_parameters([0.0, 0.0])
        return False  # failed to turn away from the obstacle

class OpenDoor(Skill):
    '''
    opens a door by moving forward until the door is detected by the left lidars and not the front ones.
    '''
    def __init__(self, lidar_front_left: LidarSensor, lidar_front_right: LidarSensor, lidar_left: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float, timeout: float = 10.0):
        '''
        lidar_front_left - a lidar with angle (0-30) deg
        lidar_front_right - a lidar with angle (330-360) deg
        lidar_left - a lidar with angle (60-120) deg
        wheels - a set of 2 wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity - the speed at which the robot should move
        timeout - the maximum time to attempt opening the door (in seconds)
        '''
        super().__init__()
        self.lidar_front_left = lidar_front_left
        self.lidar_front_right = lidar_front_right
        self.lidar_left = lidar_left
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.timeout = timeout

    def execute(self):
        self.wheels.set_parameters([self.velocity, self.velocity])
        start_time = time.time()
        while time.time() - start_time < self.timeout:  # give it max. timeout seconds to open the door
            if self.lidar_left.read() < self.min_dist and self.lidar_front_left.read() > self.min_dist and self.lidar_front_right.read() > self.min_dist:
                self.wheels.set_parameters([0.0, 0.0]) # halt
                return True
        self.wheels.set_parameters([0.0, 0.0])
        return False  # failed to open the door within the timeout period
