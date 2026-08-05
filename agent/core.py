# parent class for an agent that thinks

from .agent_loop import StatsAgent
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
        self.stats = []
        

    def set_communicator(self):
        # set communicator for sensors and actuators if they exist
        if self.sensors is not None:
            for sensor in self.sensors:
                sensor.set_communicator(self.bus)
        if self.actuators is not None:
            for actuator in self.actuators:
                actuator.set_communicator(self.bus)

    def add_stats(self, stats:list[StatsAgent]):
        self.stats = stats + self.stats

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
            'Approach': Approach(self.lidars.sensors[0], self.lidars.sensors[7], self.wheels, velocity, min_dist, timeout=7.0),
            'TurnAway': TurnAway(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.lidars.sensors[5], self.wheels, velocity, min_dist),
            'OpenDoor': OpenDoor(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.wheels, velocity, min_dist, timeout=4.0)
            }
        self.brain = BrainNetwork()

    def run(self):
        try:
            for step in range(100):
                # 0. Update statistics
                for s in self.stats:
                    s.update()
                # 1. Read State
                prev_state_id, prev_state_vector = self.read_state()
                # 2. Create PDDL -- to be implemented --

                # 3. Make a Plan -- to be implemented --
                # 4. Execute Next Step -- to be implemented --
                # 5.A Explore
                self.explore()
                # 5.B Read State  -- to be implemented --
                # 6. Update Network  -- to be implemented --
        finally:
            for s in self.stats:
                s.close()
                
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
        while True:
            # read state
            prev_state_id, prev_state_vector = self.read_state() 
            # execute random action
            skill = random.choice(list(self.skillset.values()))
            print(f"{skill.__class__.__name__} executed... ") # DEBUG
            success = skill.execute()   # successfully?
            print("nicely") if success else print("poorly")
            if success: break           # YES = go on
                                        # NO = back to the begining
        # read new state
        new_state_id, new_state_vector = self.read_state() 
        # could we predict it using transitional network?
        preddicted_state_id = self.brain.predict(prev_state_id, skill)
        if new_state_id == preddicted_state_id: # are we in correct state?
            return                              # YES = nothing new to learn
        else:                                   # NO = we have to add this to our brain
            self.brain.update(prev_state_vector, skill, new_state_vector)
            return

        
    def create_pddl(self):
        pass

    def make_a_plan(self):
        pass

    def execute_next_step(self):
        pass

    def update_networks(self):
        pass
