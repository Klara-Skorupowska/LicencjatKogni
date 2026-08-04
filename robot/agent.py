# parent class for an agent that thinks

from communicator import Communicator
from .virtual_actuator import *
from .virtual_sensor import *
from .skills import *
from .brain_network import *

import cv2
import numpy as np
import random

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
    it just avoids running into walls but uses skills. at the start it tests some skills
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
            'MoveForward': MoveForward(self.lidars.sensors[0], self.lidars.sensors[7], self.wheels, velocity, min_dist, time=1.0),
            'TurnRight': TurnRight(self.lidars.sensors[0], self.lidars.sensors[7], self.wheels, velocity, min_dist, time=1.0),
            'TurnLeft': TurnLeft(self.lidars.sensors[0], self.lidars.sensors[7], self.wheels, velocity, min_dist),
            'TurnAway': TurnAway(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.lidars.sensors[5], self.wheels, velocity, min_dist),
            'OpenDoor': OpenDoor(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.wheels, velocity, min_dist, timeout=7.0)
            }
        
    def read_state(self):
        '''Read the current state of the agent and return if there is a danger in front of us - two possible states '''
        # are we in danger? we have to duck if there is anything in front of us:
        duck = False
        self.lidars.read()
        self.cctv.read() # just to update the camera frame, we don't use it here
        for sensor in self.lidars.sensors:
            if (sensor.lidar_direction < 45 or sensor.lidar_direction > 315) and sensor.value < 0.07:
                duck = True
                break
        return duck

    def run(self):
        self.read_state()
        if not self.skillset['OpenDoor'].execute():
            print("OpenDoor failed :(")
        else:
            print("OpenDoor succeeded :)")
        if not self.skillset['TurnRight'].execute():
            print("TurnRight failed :(")
        else:
            print("TurnRight succeeded :)")
        if not self.skillset['TurnLeft'].execute():
            print("TurnLeft failed :(")
        else:            
           print("TurnLeft succeeded :)")    

        while True:
            duck = self.read_state()
            if duck:
                if not self.skillset['TurnAway'].execute():
                    print("TurnAway failed :(")
                else: 
                    print("TurnAway succeeded :)")
            else:
                if not self.skillset['MoveForward'].execute():
                    print("MoveForward failed :(")
                else: 
                    print("MoveForward succeeded :)")


class TheAgent(Agent):
    '''
    it explores the environment and abstracts symbols from used skills. it also accomplish the goal of moving from one pad to another.
    '''
    def __init__(self, bus: Communicator):
        super().__init__(bus)
        # Initialize sensors and actuators here
        self.wheels = WheelsActuator(self.bus)
        self.lidars = VirtualSensorArray(self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]])
        self.camera = CameraSensor(self.bus)
            # common parameters:
        min_dist = 0.05
        velocity = 10
        self.skillset = {
            'MoveForward': MoveForward(self.lidars.sensors[0], self.lidars.sensors[7], self.wheels, velocity, min_dist, time=1.0),
            'TurnAway': TurnAway(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.lidars.sensors[5], self.wheels, velocity, min_dist)}
            #'OpenDoor': OpenDoor(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.wheels, velocity, min_dist, timeout=7.0)
            #}
        self.brain = BrainNetwork()

    def run(self):
        for step in range(1000):
            # A. Observe the environment BEFORE moving
            prev_state_id, prev_state_vector = self.read_state()
    
            # B. Choose a skill to execute (you could randomize this for exploration)

            # skill = random.choice(list(self.skillset.values()))
            duck = self.duck()
            if duck:
                skill = self.skillset['TurnAway']
            else:
                skill = self.skillset['MoveForward']
    
            # C. Execute the skill (this blocks until it finishes or times out)
            success = skill.execute()
    
            # D. Observe the environment AFTER moving
            current_state_id, current_state_vector = self.read_state()
    
            # E. Update the Brain!
            # This automatically updates the GNG nodes and the NetworkX transitional map.
            self.brain.update(prev_state_vector, skill, current_state_vector)
    
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"Step {step}: Brain has {len(self.brain.gng_nodes)} topological nodes and {len(self.brain.transitional_map.edges)} mapped transitions.")
                
    def duck(self):
        duck = False
        self.lidars.read()
        for sensor in self.lidars.sensors:
            if (sensor.lidar_direction < 45 or sensor.lidar_direction > 315) and sensor.value < 0.07:
                duck = True
                break
        return duck

    def read_state(self):
        # 1. read sensors
        self.lidars.read()
        self.camera.read()
        # 2. preprocess camera data
        self.camera.preprocess()
        # 3. evaluate symbols (not implemented yet)
        # ...
        # 4. classify state using brain network
        # Combine standard Python lists, then make it a NumPy array
        state_vector = np.array(self.lidars.value + self.camera.coded, dtype=float)
        self.brain._update_scaling(state_vector)
        state_id = self.brain.classify(state_vector)
        return state_id, state_vector

    def explore(self):
        pass

    def create_pddl(self):
        pass

    def make_a_plan(self):
        pass

    def execute_next_step(self):
        pass

    def update_networks(self):
        pass
