# parent class for an agent that thinks
from communicator import Communicator
from .virtual_actuator import *
from .virtual_sensor import *
from .skills import *
from .skills_serial import *
from .brain_network import *

import cv2
import numpy as np
import random
import os
import subprocess
import sys

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
        self.lidars = VirtualSensorArray(self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]])
        self.cctv = CameraSensor(self.bus)
        
    def read_state(self):
        distances = [sensor.read() for sensor in self.lidars.sensors]
        # are we in danger?
        duck = (min(distances) != max(distances))
        # Where is nearest obstacle?
        idx = distances.index(min(distances))
        ang = self.lidars.sensors[idx].lidar_direction
        dist = self.lidars.sensors[idx].value
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
    skill testing agent.
    '''
    def __init__(self, bus: Communicator):
        super().__init__(bus)
        # Initialize sensors and actuators here
        self.wheels = WheelsActuator(self.bus)
        self.lidars = VirtualSensorArray(self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]])
        self.camera = CameraSensor(self.bus)
            # common parameters:
        min_dist = 0.05
        velocity = 15
        self.skillset = {
            'SpotTheDoor': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, min_dist, timeout=5.0),
            'GoToTheDoor 2.1': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, min_dist=min_dist, timeout=15.0),
            'GoThroughTheDoor': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, min_dist, timeout=15.0),
            'SpotTheGoal': SpotTheColor(120, self.camera, self.lidars, self.wheels, velocity, min_dist, timeout=5.0),
            'GoToTheGoal 2.1': GoToTheGoal(bus, 120, self.camera, self.lidars, self.wheels, velocity, min_dist=min_dist, timeout=15.0),
            'ClearThePath': ClearThePath(bus, self.camera, self.lidars, self.wheels, velocity, min_dist=min_dist, run_time=2.0, timeout=15.0),
        }
        
    def read_state(self):
        '''Read the current state of the agent and return if there is a danger in front of us - two possible states '''
        # are we in danger? we have to duck if there is anything in front of us:
        duck = False
        self.lidars.read()
        self.camera.read() # just to update the camera frame, we don't use it here
        for sensor in self.lidars.sensors:
            if (sensor.lidar_direction < 45 or sensor.lidar_direction > 315) and sensor.value < 0.07:
                duck = True
                break
        return duck

    def run(self):
        self.skillset["ClearThePath"].execute()
        while True:
            duck = self.read_state()
            if duck:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Oh noo, tha wall...")

            time.sleep(1.0)
            for name, skl in self.skillset.items():
                if not skl.execute():
                    print(f" {name} failed.")
                    time.sleep(1.0)
            