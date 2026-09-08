import enum
from tkinter import ACTIVE

from .core import Agent
from communicator import Communicator
from .virtual_actuator import *
from .virtual_sensor import *
from .skills_serial import *
from .brain_network import *

import os
import sys
import subprocess
import random
import numpy as np
from datetime import datetime
import time

class TheAgent(Agent):
    '''
    It explores the environment and abstracts symbols from used skills. 
    It also accomplishes the goal of moving from one pad to another.
    '''
    def __init__(self, bus: Communicator):
        super().__init__(bus)
        self.wheels = WheelsActuator(self.bus)
        self.lidars = VirtualSensorArray(self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]])
        self.camera = CameraSensor(self.bus)
        
        self.max_runs = 100 # ~ how many explore actions we agreed on

        self.fail_buffor = []
        self.buffor_max_len = 10 

        self.plan_count = 0 

        save_dist = 0.05
        velocity = 15
        self.skillset = {
            'SpotTheDoor': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=3.0),
            'GoToTheDoor': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'GoThroughTheDoor': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=10.0),
            'SpotTheGoal': SpotTheColor(120, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=3.0),
            'GoToTheGoal': GoToTheGoal(bus, 120, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'Finnish': Finnish(bus, 120, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
        }
        
        # 1. Directory Structure Setup
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join("logs", timestamp)
        self.pddl_dir = os.path.join(self.log_dir, "PDDL")
        self.stats_path = os.path.join(self.log_dir, "execution_statistics.txt")
        
        # 2. Add brain and pass it the root log directory
        self.brain = BrainNetwork(log_dir=self.log_dir)

        # 3. Setup Skill Execution Logger
        self.skill_stats = {name: {'success': 0, 'fail': 0} for name in self.skillset.keys()}
        self.start_time = None
        self.end_time = None
        self.run_count = 0
        self.update_timestamps = []

    def _logger(self, run):
        if not self.stats_path: return
        if self.end_time is None: self.end_time = time.time()
        with open(self.stats_path, "w") as f:
            start_str = datetime.fromtimestamp(self.start_time).strftime("%d-%m-%Y %H:%M:%S")
            end_str = datetime.fromtimestamp(self.end_time).strftime("%d-%m-%Y %H:%M:%S")
                
            duration = self.end_time - self.start_time
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
            f.write(f"start time: {start_str} | end time: {end_str} | duration: {duration_str}\n")
            f.write(f"run: {run}; brain updates: {len(self.update_timestamps)}\n")
            f.write(f"{'Skill Name':<25} | {'Successes':<10} | {'Fails':<10}\n")
            f.write("-" * 55 + "\n")
            for sk_name, counts in self.skill_stats.items():
                sumall = counts['success'] + counts['fail']
                f.write(f"{sk_name:<25} | {counts['success']:<10} | {counts['fail']:<10} | {sumall:<10}\n")
            # todo: update timestamps

    def run(self):
        try:
           clear_run = False
           self.run_count = 0
           self.start_time = time.time()
           prev_skill_name = None
           while not clear_run:
               if self.run_count > self.max_runs: 
                   print(f"[Agent] I give up.")
                   break
               clear_run = True
               self._logger(self.run_count)
               self.run_count += 1
               print(f"[Agent] Run {self.run_count}/{self.max_runs} ---------------")
               #-> Generate Plan
               plan = [] #self.generate_plan() # TODO
               if len(plan) < 1: # what if plan is empty? Explore
                   print("[Agent] Returned empty plan.")
                   clear_run = False
                   prev_skill_name = self.explore(prev_skill_name) 
                   continue
               #-> Execute Next Step of a Plan
               print("[Agent] Start executing plan.")
               for i, skill_name in enumerate(plan):
                   print(f"[Agent] Executing step {i}/{len(plan)}: {skill_name}")
                   success = self.execute_skill(skill_name, prev_skill_name)
                   #-> Was it successful?
                   if not success: #-> if not, explore
                       clear_run = False
                       prev_skill_name = self.explore(prev_skill_name)
                       continue
                   #-> if yes, do the next step of the plan
                   prev_skill_name = skill_name

        finally:
            print("[Agent] Clear run completed.")
            self.end_time = time.time()
            self.brain._logger()
            self._logger(self.run_count-1)
            
               
    def execute_skill(self, skill_name, prev_skill_name):
        print(f"[Agent] Execute Skill")
        #-> read state
        prev_vector_state = self.read_state()
        #-> Are preconditions of this skill met?
        preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state)
        #-> execute the skill
        skill = self.skillset[skill_name]
        success = skill.execute()
        self.skill_stats[skill_name]['success' if success else 'fail'] += 1 # collect skill execution stats
        vector_state = self.read_state()
        #-> Is the outcome expectable?
        expectable = (preconditions_are_met and success) or (not preconditions_are_met and not success)
        if not expectable: # if not
            #-> Add unexpected data to buffor
            data = (prev_vector_state, skill_name, vector_state, success)
            self.add_to_buffor(data)
        else: # if yes
            #-> Did we finnished? If so, add it to the data
            if not skill_name == 'Finnish':
                self.execute_skill('Finnish', skill_name)
        return success

    def read_state(self):
        print("[Agent] Read State")
        #-> read sensors
        self.lidars.read()
        self.camera.read()
        #-> preprocess raw data, vision + scaling 0-1
        self.camera.preprocess()
        self.lidars.preprocess()

        #-> return state
        state_vector = np.concatenate([self.lidars.coded, self.camera.coded]).flatten()
        return state_vector

    def explore(self, prev_skill_name):
        print("[Agent] Explore")
        #-> choose random action
        available_skills = [k for k in self.skillset if k != "Finnish"]
        if not available_skills:
            return None
        skill_name = random.choice(available_skills)
        #-> execute action
        self.execute_skill(skill_name, prev_skill_name)
        #-> return this action
        return skill_name
     
    def add_to_buffor(self, data):
        if len(self.fail_buffor) >= self.buffor_max_len:
            self.brain.update(self.fail_buffor)
            self.update_timestamps += [time.time()]
            self.fail_buffor = []
        self.fail_buffor.append(data)


    def generate_plan(self):
        print("[Agent] Generate Plan")
        #-> read state
        vector_state = self.read_state()
        #-> get active skills: by resolving predicates, which skills are doable right now
        active_skills = self.brain.get_active_skills(vector_state)
        if not active_skills: # if no active skills (nodes), no plan at all
            return [] 
        #-> choose the starting skill == the node
        start_skill = random.choice(active_skills)
        #-> choose the goal
        found_goal = False
        nodes = self.brain.get_nodes()
        if not nodes:
            return []
        for node in nodes:
            if node == 'Finnished':
                found_goal = True
                goal_skill = node
                break
        if not found_goal:
            goal_skill = random.choice(nodes)

        # the paths
        self.plan_count += 1
        plan_dir = os.path.join(self.pddl_dir, f"plan_{self.plan_count}")
        domain_file = os.path.join(plan_dir, f"domain.pddl")
        problem_file = os.path.join(plan_dir, f"problem.pddl")
        plan_file = os.path.join(plan_dir, f"problem.pddl.soln")

        #-> generate domain.pddl
        domain_str = self.brain.generate_domain_pddl()
        with open(domain_file, "w") as f:
            f.write(domain_str)
        #-> generate problem.pddl
        problem_str = self.brain.generate_problem_pddl(initial_state=start_skill, goal_state=goal_skill)
        with open(problem_file, "w") as f:
            f.write(problem_str)
        #-> run solver
        if not os.path.exists(domain_file) or not os.path.exists(problem_file):
            print(f"[Agent] Error: no .pddl files")
            return []

        if os.path.exists(plan_file):
            os.remove(plan_file)

        try:
            res = subprocess.run(
                [sys.executable, "-m", "pyperplan", domain_file, problem_file],
                capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            # Print compiler errors to catch missing objects or domain syntax issues
            print(f"[Agent] Pyperplan error: {e.stderr or e.stdout}")
            return []

        if not os.path.exists(plan_file):
            return []

        #-> change to executable plan
        executable_plan = []
        with open(plan_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue 

                # Format is: (skill_name source_node target_node) ---------------- TODO ----------------
                parts = line.strip("()").split()
                if len(parts) < 3:
                    continue
            
                action_name_lower = parts[0].lower()
                target_state = parts[2].replace("node_", "")

                matched_skill = next(
                    (sk for sk in self.skillset.keys() if sk.lower() == action_name_lower), 
                    None
                )
                if matched_skill:
                    executable_plan.append((matched_skill, target_state))
        if len(executable_plan)==0:
            print(f"[Agent] Error: no executable plan.")
        return executable_plan
