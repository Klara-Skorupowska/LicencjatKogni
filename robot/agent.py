# parent class for an agent that thinks

from tkinter import SE
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
    it just avoids running into walls but uses skills
    '''
    def __init__(self, bus: Communicator):
        super().__init__(bus)
        # Initialize sensors and actuators here
        self.wheels = WheelsActuator(self.bus)
        self.lidars = VirtualSensorArray(self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]])
        self.cctv = CameraSensor(self.bus)
            # common parameters:
        min_dist = 0.05
        velocity = 10
        self.skillset = {
            'MoveForward': MoveForward(self.lidars.sensors[0], self.lidars.sensors[7], self.wheels, velocity, min_dist),
            'TurnRight': TurnRight(self.lidars.sensors[0], self.lidars.sensors[7], self.wheels, velocity, min_dist),
            'TurnLeft': TurnLeft(self.lidars.sensors[0], self.lidars.sensors[7], self.wheels, velocity, min_dist),
            'TurnAway': TurnAway(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.lidars.sensors[5], self.wheels, velocity, min_dist),
            'OpenDoor': OpenDoor(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.wheels, velocity, min_dist, timeout=10.0)
            }
        
    def read_state(self):
        '''Read the current state of the agent and return if there is a danger in front of us, at what angle and distance '''
        distances = [sensor.read() for sensor in self.lidars.sensors]
        # are we in danger? we have to duck if there is anything in front of us:
        duck = False
        ang = None
        dist = None
        for sensor in self.lidars.sensors:
            if (sensor.lidar_direction < 45 or sensor.lidar_direction > 315) and sensor.value < 0.07:
                duck = True
                ang = sensor.lidar_direction
                dist = sensor.value
                break
        return duck, ang, dist

    def run(self):
        while True:
            duck, _, _= self.read_state()
            self.cctv.read()
            if duck:
                if not self.skillset['TurnAway'].execute():
                    print("TurnAway failed :(")
            else:
                self.skillset['MoveForward'].execute()

