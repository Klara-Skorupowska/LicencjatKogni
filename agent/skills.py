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

    def execute(self):
        ''' executes the skill from start to finish and returns if the execution was completed '''
        raise NotImplementedError("The execute() method must be implemented in the subclass.") 



### --- EXTRA CALCULATIONS FOR EPUCK2   --- ###
T_CONST = math.pi*0.052/(0.02)  # based on epuck2 (wheel radius and distance between them), needed to be devided by the velocity to get the time needed to turn 360 deg
### ---                                 --- ###
class MoveForward(Skill):
    ''' moves forward until it aproaches a wall '''
    def __init__(self, lidar_forward_left: LidarSensor, lidar_forward_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float, time: float = 5.0):
        '''
        lidar_forward_left - a lidar with angle (0-30) deg
        lidar_forward_right - a lidar with angle (330-360) deg
        wheels - a set of wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity 
        time - the time of moving forward (in seconds)
        '''
        super().__init__()
        self.lidar_forward_left = lidar_forward_left
        self.lidar_forward_right = lidar_forward_right
        self.wheels = wheels

        self.velocity = velocity
        self.min_dist = min_dist
        self.time = time

    def execute(self):
        self.wheels.set_parameters([self.velocity, self.velocity])
        start_time = time.time()
        
        while time.time() - start_time < self.time:
            # 1. Take a snapshot of the sensors
            front_left = self.lidar_forward_left.read()
            front_right = self.lidar_forward_right.read()

            # 2. Check if we have approached a wall
            if front_left < self.min_dist or front_right < self.min_dist:
                self.wheels.set_parameters([0.0, 0.0]) # halt
                return True # completed moving forward until hitting a wall
                
            # 3. Yield to the CPU
            time.sleep(0.01)
            
        # If we reach this point, the timeout expired before finding a wall = no need for halting, as we are still moving forward
        return True  # completed moving forward without hitting a wall

class TurnLeft(Skill):
    '''
    left turn for set time
    '''
    def __init__(self, lidar_forward_left: LidarSensor, lidar_forward_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float, time: float = 1.0):
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
        self.time = time

    def execute(self):
        start_time = time.time()
        while time.time() - start_time < self.time:
            self.wheels.set_parameters([-self.velocity, self.velocity])
            time.sleep(0.01)  # Prevent CPU spinning

        self.wheels.set_parameters([0.0, 0.0]) # halt
        return True

class TurnRight(Skill):
    '''
    right turn for set time
    '''
    def __init__(self, lidar_forward_left: LidarSensor, lidar_forward_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float, time: float = 1.0):
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
        self.time = time

    def execute(self):
        start_time = time.time()
        while time.time() - start_time < self.time:
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
        # 1. Find nearest obstacle
        front_left= self.lidar_front_left.read()
        front_right = self.lidar_front_right.read()

        if front_left < front_right:
            turn_direction = 'right'
            target_distance = front_left
            # If turning right (clockwise), the obstacle slides down the left side (CCW relative to robot)
            front_tracker = self.lidar_front_left
            back_tracker = self.lidar_back_left
        else:
            turn_direction = 'left'
            target_distance = front_right
            # If turning left (CCW), the obstacle slides down the right side (CW relative to robot)
            front_tracker = self.lidar_front_right
            back_tracker = self.lidar_back_right

        # 2. Initiate turn
        if turn_direction == 'right':
            self.wheels.set_parameters([self.velocity, -self.velocity])
        else:
            self.wheels.set_parameters([-self.velocity, self.velocity])

        full_circle_time = T_CONST / self.velocity
        epsilon = 0.01
        start_time = time.time()
        
        front_cleared = False 

        while time.time() - start_time < full_circle_time:
            # Snapshot required lidars
            current_front = front_tracker.read()
            current_back = back_tracker.read()
            
            current_front_left = self.lidar_front_left.read()
            current_front_right = self.lidar_front_right.read()

            # Step A: Check if we have rotated enough for the obstacle to leave the front sensor
            if not front_cleared:
                if abs(current_front - target_distance) >= epsilon:
                    front_cleared = True

            # Step B: If front is clear, check if the obstacle has reached the back sensor
            obstacle_is_behind = False
            if front_cleared:
                obstacle_is_behind = abs(current_back - target_distance) <= epsilon
                
            # Step C: Ensure our new path is safe
            path_is_clear = (current_front_left > self.min_dist) and (current_front_right > self.min_dist)

            # Step D: Exit condition
            if obstacle_is_behind and path_is_clear:
                self.wheels.set_parameters([0.0, 0.0]) # halt
                return True
                
            time.sleep(0.01)

        # Timeout reached (completed 360 turn without finding a clear path)
        self.wheels.set_parameters([0.0, 0.0])
        return False

class OpenDoor(Skill):
    '''
    Opens a door by moving forward until the door is detected by the left lidars 
    and not the front ones.
    '''
    def __init__(self, 
                 lidar_front_left: LidarSensor, 
                 lidar_front_right: LidarSensor, 
                 lidar_left: LidarSensor, 
                 wheels: WheelsActuator, 
                 velocity: float, 
                 min_dist: float, 
                 timeout: float = 10.0):
        super().__init__()
        self.lidar_front_left = lidar_front_left
        self.lidar_front_right = lidar_front_right
        self.lidar_left = lidar_left
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.timeout = timeout

    def execute(self):
        # Start moving forward
        self.wheels.set_parameters([self.velocity, self.velocity])
        
        front_left_dist = self.lidar_front_left.read()
        front_right_dist = self.lidar_front_right.read()

        if front_left_dist > self.min_dist and front_right_dist > self.min_dist:
            return False  # No door detected in front, cannot open

        start_time = time.time()
        while time.time() - start_time < self.timeout:
            # 1. Take a simultaneous snapshot of all required sensors
            left_dist = self.lidar_left.read()
            front_left_dist = self.lidar_front_left.read()
            front_right_dist = self.lidar_front_right.read()
            
            # 2. Evaluate the snapshot
            door_on_left = left_dist < self.min_dist
            front_is_clear = (front_left_dist > self.min_dist) and (front_right_dist > self.min_dist)
            
            if door_on_left and front_is_clear:
                self.wheels.set_parameters([0.0, 0.0]) # halt
                return True # there was a door and we opened it successfully
                
            # 3. Yield to the CPU
            time.sleep(0.01)
            
        # Timeout reached
        self.wheels.set_parameters([0.0, 0.0])
        return False # there was a door but we failed to open it in time


class Approach(Skill):
    ''' moves forward until it runs into an obstacle (or out of time) and position itself directly in front of it '''
    def __init__(self, lidar_front_left: LidarSensor, lidar_front_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float, timeout: float, epsilon: float = 0.001):
        '''
        lidar_front_left - a lidar with angle (0-30) deg
        lidar_front_right - a lidar with angle (330-360) deg
        wheels - a set of 2 wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity - the speed at which the robot should move
        timeout - time after which robot stops and the aproach fails
        epsilon - acceptable distance error
        '''
        super().__init__()
        self.lidar_front_left = lidar_front_left
        self.lidar_front_right = lidar_front_right
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.timeout = timeout
        self.epsilon = epsilon

    def execute(self):
        # Start moving forward
        self.wheels.set_parameters([self.velocity, self.velocity])
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            # 1. Take a simultaneous snapshot of all required sensors
            front_left_dist = self.lidar_front_left.read()
            front_right_dist = self.lidar_front_right.read()
            
            # 2. Evaluate the snapshot
            if front_left_dist <= self.min_dist or front_left_dist <= self.min_dist:  # if there is an obstacle ahead
                # - start to position itself in front of it = turn -
                if front_left_dist >= front_right_dist:
                    self.wheels.set_parameters([self.velocity/2, -self.velocity/2])
                else:
                    self.wheels.set_parameters([-self.velocity/2, self.velocity/2])
                start_time = time.time() # timeout for positioning
                while time.time() - start_time < self.timeout:
                    # snapshot of sensors
                    front_left_dist = self.lidar_front_left.read()
                    front_right_dist = self.lidar_front_right.read()
                    # check if we are facing the wall
                    if abs(front_right_dist - front_left_dist) < self.epsilon: 
                        self.wheels.set_parameters([0.0, 0.0])
                        return True # success: in front of the wall
                    time.sleep(0.01)
            # 3. Yield to the CPU
            time.sleep(0.01)
            
        # Timeout reached
        self.wheels.set_parameters([0.0, 0.0])
        return False # there was no obstacle that we could reach in time
