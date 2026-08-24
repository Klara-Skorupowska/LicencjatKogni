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
import os
import subprocess
import sys

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
            'TurnRight': Turn('right', self.wheels, velocity, time=1.0),
            'TurnLeft': Turn('left', self.wheels, velocity, time=1.0),
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
        # Initialize sensors and actuators
        self.wheels = WheelsActuator(self.bus)
        self.lidars = VirtualSensorArray(self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]])
        self.camera = CameraSensor(self.bus)
        # Prepare the skill set
            # common parameters:
        min_dist = 0.05
        velocity = 10
        self.skillset = {
            'Approach': Approach(self.lidars.sensors[0], self.lidars.sensors[7], self.wheels, velocity, min_dist, timeout=10.0),
            'TurnAway': TurnAway(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.lidars.sensors[5], self.wheels, velocity, min_dist),
            'TurnLeft': Turn('left', self.wheels, velocity, 1.0),
            'TurnRight': Turn('right', self.wheels, velocity, 1.0),
            'OpenDoor': OpenDoor(self.lidars.sensors[0], self.lidars.sensors[7], self.lidars.sensors[2], self.wheels, velocity, min_dist, timeout=10.0)
            }
        # add a brain
        self.brain = BrainNetwork()
        # are we done?
        self.finnished = FinnishSensor(bus)
        # PDDL files
        self.domain_path = "pddl/domain.pddl"
        self.problem_path = "pddl/problem.pddl"
        self.plan_file = "pddl/problem.pddl.soln"

    def run(self):
        try:
            n = 50
            for step in range(n):
                print(f"[Agent] Step {step}/{n}")
                # 0. Update statistics
                for s in self.stats:
                    s.update()

                # 1. Read State
                prev_state_id, prev_state_vector = self.read_state()

                # check for goal (maybe we are already there)
                self.finnished.read()
                if self.finnished.value:
                    self.brain.transitional_map.nodes[prev_state_id]['is_goal'] = True

                # Check if we know the goal to initiate PDDL
                goal_node_id = None
                for node_id, node_data in self.brain.transitional_map.nodes(data=True):
                    if node_data.get('is_goal', False):
                        goal_node_id = node_id
                        break
                        
                if goal_node_id is None:
                    # If goal is not yet known, fallback to exploration
                    print("[Agent] Goal not discovered yet. Continuing exploration...")
                    self.explore()
                    continue

                # Inner loop to handle the "Update Networks -> Create PDDL" flowchart cycle
                while True:
                    # 2. Create PDDL
                    self.create_pddl(prev_state_id, goal_node_id)
                    
                    # 3. Make a Plan
                    plan = self.make_plan()
                    
                    if not plan:
                        self.explore()
                        break # Break out to Main Loop -> Read State
                    
                    needs_explore = False
                    needs_replan = False
                    
                    # Execute Plan
                    for step_idx, (skill_name, target_state_str) in enumerate(plan):
                        skill = self.skillset[skill_name]
                        expected_state_id = int(target_state_str.replace('s', ''))
                        
                        # 4. Execute Next Step
                        success = skill.execute()
                        
                        # Decision: Successfully?
                        if not success:
                            # no -> Explore -> Read State
                            self.explore()
                            needs_explore = True
                            break # Break to Main Loop
                            
                        # yes -> Read State
                        new_state_id, new_state_vector = self.read_state()
                        
                        # Decision: Correct State?
                        if new_state_id != expected_state_id:
                            # no -> Update Networks -> Create PDDL
                            self.brain.update(prev_state_vector, skill, new_state_vector)
                            prev_state_id = new_state_id
                            prev_state_vector = new_state_vector
                            needs_replan = True
                            break # Break to Create PDDL loop
                            
                        # yes -> Decision: Last Step?
                        is_last_step = (step_idx == len(plan) - 1)
                        if not is_last_step:
                            # no -> Execute Next Step
                            prev_state_id = new_state_id
                            prev_state_vector = new_state_vector
                            continue
                            
                        # yes -> Decision: Final State?
                        self.finnished.read()
                        if not self.finnished.value:
                            # no -> Update Networks -> Create PDDL
                            self.brain.update(prev_state_vector, skill, new_state_vector)
                            prev_state_id = new_state_id
                            prev_state_vector = new_state_vector
                            needs_replan = True
                            break # Break to Create PDDL loop
                        else:
                            # yes -> Finnish
                            print("[Agent] Successful plan noted.!")
                            with open("pddl/successful_plan.txt", "w") as f:
                                f.write("Successful Plan Executed:\n")
                                for p_step, (p_skill, p_target) in enumerate(plan, 1):
                                    f.write(f"Step {p_step}: {p_skill} -> {p_target}\n")
                            return
                    
                    if needs_explore:
                        break # Break out to Main Loop -> Read State
                    if needs_replan:
                        continue # Loop back to -> Create PDDL
            print("[Agent] The End")
                
        finally:
            self.finnished.read()
            if self.finnished.value:
                print("[Agent] Successfully reached goal")
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
        # 3. classify state using brain network
        # Combine standard Python lists, then make it a NumPy array
        state_vector = np.array(self.lidars.value + self.camera.coded, dtype=float)
        self.brain._update_scaling(state_vector)
        state_id = self.brain.classify(state_vector)
        return state_id, state_vector

    def explore(self):
        '''
        execute random action, update nets if it brings new info
        '''
        print("[Agent] Exploring...")
        while True:
            # read state
            prev_state_id, prev_state_vector = self.read_state()
            
            # check for goal (maybe we are already there)
            self.finnished.read()
            if self.finnished.value:
                self.brain.transitional_map.nodes[prev_state_id]['is_goal'] = True

            # execute random action
            skill = random.choice(list(self.skillset.values()))

            success = skill.execute()   # successfully?
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
        
    
    def create_pddl(self, start_node_id: int, goal_node_id: int):
        '''
        Uses on-demand aggregation of the continuous GNG to create PDDL files 
        with a plan to reach the end pad.
        '''
        # Ensure target directory exists as requested
        os.makedirs('pddl', exist_ok=True) 

        # 1. On-Demand Aggregation: Snapshot the current state of the GNG
        # This returns the dict with 'min_bound', 'max_bound', and 'raw_nodes'
        symbolic_bounds = self.brain.get_symbolic_representation()

        # 2. Extract unique skills and abstract states to build the PDDL strings
        # A state in this context is simply the destination node ID from the GNG
        skills = set()
        states = set([start_node_id, goal_node_id]) 
        for category in ['preconditions', 'effects']:
            for symbol_name, data in symbolic_bounds[category].items():
                # symbol_name looks like: "Pre_MoveForward_to_5" or "Eff_TurnRight_from_2"
                skill = data['skill']
                node = data['node']
        
                skills.add(skill)
                states.add(node)

                # Write the continuous bounding boxes for the robot's physical execution
                with open(f"pddl/{symbol_name}.txt", "w") as f:
                    f.write(f"MIN: {data['min'].tolist()}\n")
                    f.write(f"MAX: {data['max'].tolist()}\n")

        # 3. Generate domain.pddl
        domain_str = "(define (domain robot-skills)\n"
        domain_str += "  (:requirements :typing)\n"
        domain_str += "  (:types state)\n"
        domain_str += "  (:predicates\n"
    
        # Define explicit Precondition and Effect predicates for every known skill
        for skill in skills:
            domain_str += f"    (Pre-{skill} ?s - state)\n"
            domain_str += f"    (Eff-{skill} ?s - state)\n"
        domain_str += "    (robot-at ?s - state)\n  )\n\n"

        # Define the actions logically requiring the Pre- and causing the Eff- symbols
        for skill in skills:
            domain_str += f"  (:action execute-{skill}\n"
            domain_str += f"    :parameters (?from - state ?to - state)\n"
            domain_str += f"    :precondition (and (robot-at ?from) (Pre-{skill} ?to))\n"
            domain_str += f"    :effect (and (not (robot-at ?from)) (robot-at ?to) (Eff-{skill} ?to))\n"
            domain_str += "  )\n"
        domain_str += ")"

        with open(self.domain_path, "w") as f:
            f.write(domain_str)

        # 4. Generate problem.pddl
        problem_str = "(define (problem reach-end-pad)\n"
        problem_str += "  (:domain robot-skills)\n"
        problem_str += "  (:objects\n    "
        problem_str += " ".join([f"s{s}" for s in states]) + " - state\n  )\n"
    
        problem_str += "  (:init\n"
        problem_str += f"    (robot-at s{start_node_id})\n"

        # Map the existing topological edges to the symbolic preconditions
        # This implicitly respects the continuous node clusters without complex overlapping logic
        for category in ['preconditions', 'effects']:
            for symbol_name in symbolic_bounds[category].keys():
                parts = symbol_name.split('_')
                prefix = parts[0]
                skill = data['skill']
                node = data['node']
        
                if prefix == "Pre":
                    problem_str += f"    (Pre-{skill} s{node})\n"
                elif prefix == "Eff":
                    problem_str += f"    (Eff-{skill} s{node})\n"
            
        problem_str += "  )\n"
        problem_str += "  (:goal\n"
        problem_str += f"    (robot-at s{goal_node_id})\n"
        problem_str += "  )\n)"

        with open(self.problem_path, "w") as f:
            f.write(problem_str)
        
        print(f"[Agent] Generated on-demand PDDL for {len(states)} abstract states.")

    def make_plan(self):
        ''' 
        Reads the generated PDDL files, calls a local pyperplan solver, 
        and translates the output into a list of executable skill tuples.
        '''

        # Fast-Failing: Ensure the files exist before wasting compute
        if not os.path.exists(self.domain_path) or not os.path.exists(self.problem_path):
            print("[Planner] Missing PDDL files. Cannot generate a plan.")
            return []

        # Clean up any old solution file so we don't accidentally read stale data
        if os.path.exists(self.plan_file):
            os.remove(self.plan_file)

        # 1. Execute the local Pyperplan solver
        try:
            print("[Planner] Querying local Pyperplan solver...")
            # subprocess.run blocks the thread safely until the solver finishes
            subprocess.run(
                [sys.executable, "-m", "pyperplan", self.domain_path, self.problem_path],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError:
            # Pyperplan returns a non-zero exit code if no path to the goal exists
            print("[Planner] Solver failed to find a valid path to the goal.")
            return []
        except FileNotFoundError:
            print("[Planner] Pyperplan not found. Did you run 'pip install pyperplan'?")
            return []

        # 2. Check if the planner successfully created the solution file
        if not os.path.exists(self.plan_file):
            print("[Planner] Goal is unreachable. Returning empty plan.")
            return []

        # 3. Parse the output and map back to executable skills
        executable_plan = []
    
        with open(self.plan_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines or Pyperplan comments
                if not line or line.startswith(";"): 
                    continue 

                # Pyperplan formats steps as "(moveforward s0 s1)"
                parts = line.strip("()").split()
                if len(parts) < 3:
                    continue
            
                action_name_lower = parts[0]
                target_state = parts[2]

                # Case-Insensitive Mapping: Match the lowercased PDDL symbol 
                # back to your PascalCase skillset keys
                matched_skill = None
                for skill_name in self.skillset.keys():
                    if skill_name.lower() == action_name_lower:
                        matched_skill = skill_name
                        break

                if matched_skill:
                    # Return the tuple format required for your precondition checks
                    executable_plan.append((matched_skill, target_state))

        print(f"[Planner] Local plan found: {executable_plan}")
        return executable_plan
