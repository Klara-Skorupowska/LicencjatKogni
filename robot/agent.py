# parent class for an agent that thinks

from turtle import distance

from communicator import Communicator
from .virtual_actuator import *
from .virtual_sensor import *
from .skills import *

import time
import cv2

class Agent:
    def __init__(self, communicator: Communicator):
        self.bus = communicator
        

    def set_communicator(self):
        # set communicator for sensors and actuators if they exist
        if self.sensors is not None:
            for sensor in self.sensors:
                sensor.set_communicator(self.bus)
        if self.actuators is not None:
            for actuator in self.actuators:
                actuator.set_communicator(self.bus)


    def run(self):
        raise NotImplementedError("The run() method must be implemented in the subclass.")


class SimpleAgent(Agent):
    '''
    it just avoids running into walls
    '''
    def __init__(self, bus: Communicator):
        super().__init__(bus)
        # Initialize sensors and actuators here
        self.wheels = WheelsActuator(self.bus)
        self.lidar = [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]]
        self.cctv = CameraSensor(self.bus)
        
    def read_state(self):
        distances = [sensor.read() for sensor in self.lidar]
        # are we in danger?
        duck = (min(distances) != max(distances))
        # Where is nearest obstacle?
        idx = distances.index(min(distances))
        ang = self.lidar[idx].lidar_direction
        dist = self.lidar[idx].value
        return duck, ang, dist

    def run(self):
        while True:
            duck, ang, dist = self.read_state()
            if duck:
                if ang < 180: 
                    self.wheels.set_value(10, -5)
                else:
                    self.wheels.set_value(-5, 10)
            elif dist > 0.05:
                self.wheels.set_value(10,10)

            frame = self.cctv.read()

            cv2.imshow(f"Pilot View", frame)
            cv2.waitKey(1) 

class SkilledAgent(Agent):
    '''
    it just avoids running into walls but uses skills
    '''
    def __init__(self, bus: Communicator):
        super().__init__(bus)
        # Initialize sensors and actuators here
        self.wheels = WheelsActuator(self.bus)
        self.lidar = [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]]
        self.cctv = CameraSensor(self.bus)
        self.skillset = {
            'MoveForward': MoveForward(self.lidar[0], self.lidar[7], self.wheels, 10, 0.01),
            'TurnRight': TurnRight(self.lidar[0], self.lidar[7], self.wheels, 10, 0.01),
            'TurnLeft': TurnLeft(self.lidar[0], self.lidar[7], self.wheels, 10, 0.01)
            }
        
    def read_state(self):
        distances = [sensor.read() for sensor in self.lidar]
        # are we in danger?
        duck = (min(distances) != max(distances))
        # Where is nearest obstacle?
        idx = distances.index(min(distances))
        ang = self.lidar[idx].lidar_direction
        dist = self.lidar[idx].value
        return duck, ang, dist

    def run(self):
        while True:
            duck, ang, dist = self.read_state()
            self.skillset['TurnRight'].execute()
            time.sleep(2)
            self.skillset['TurnLeft'].execute()
            time.sleep(2)
            self.skillset['MoveForward'].execute()
            if duck:
                if ang < 180: 
                    self.skillset['TurnRight'].execute()
                else:
                    self.skillset['TurnLeft'].execute()
            else:
                self.skillset['MoveForward'].execute()

            frame = self.cctv.read()

            cv2.imshow(f"Pilot View", frame)
            cv2.waitKey(1) 
