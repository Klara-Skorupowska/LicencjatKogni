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
        
        self.max_runs = 250 # ~ how many explore actions

        self.fail_buffor = []
        self.buffor_max_len = 5 

        save_dist = 0.05
        velocity = 15
        self.skillset = {
            'SpotTheDoor': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=4.0),
            'GoToTheDoor': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'GoThroughTheDoor': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=10.0),
            'SpotTheGoal': SpotTheColor(120, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=4.0),
            'GoToTheGoal': GoToTheGoal(bus, 120, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'Finish': Finish(bus, 120, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=1.0),
        }
        
        # 1. Directory Structure Setup
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join("logs", timestamp)
        self.pddl_dir = os.path.join(self.log_dir, "PDDL")
        self.stats_path = os.path.join(self.log_dir, "execution_statistics.txt")
        self.timestamps_path = os.path.join(self.log_dir, "timestamps.txt")
        
        # 2. Add brain and pass it the root log directory
        self.brain = BrainNetwork(log_dir=self.log_dir)

        # 3. Setup Execution Logger
        self.skill_stats = {name: {'true_positive': 0, 'true_negative': 0, 'false_positive': 0, 'false_negative': 0} for name in self.skillset.keys()}
        self.start_time = None
        self.end_time = None
        self.run_count = 0
        self.batch_update_count = 0
        self.plan_count = 0
        self.timestamps = []

    def _logger(self, run):
        if self.stats_path:
            if self.end_time is None: self.end_time = time.time()
            with open(self.stats_path, "w") as f:
                start_str = datetime.fromtimestamp(self.start_time).strftime("%d-%m-%Y %H:%M:%S")
                end_str = datetime.fromtimestamp(self.end_time).strftime("%d-%m-%Y %H:%M:%S")
                
                duration = self.end_time - self.start_time
                hours, remainder = divmod(int(duration), 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                f.write(f"start time: {start_str} | end time: {end_str} | duration: {duration_str}\n")
                f.write(f"runs: {run}; brain updates: {self.batch_update_count}\n")
                f.write(f"{'Skill Name':<20} | {'True Positive':<15} | {'True Negative':<15} | {'false Positive':<15} | {'false Negative':<15} | {'Total':<15}\n")
                f.write("-" * 100 + "\n")
                for sk_name, counts in self.skill_stats.items():
                    sumall = counts['true_positive'] + counts['true_negative'] + counts['false_positive'] + counts['false_negative']
                    f.write(f"{sk_name:<20} | {counts['true_positive']:<15} | {counts['true_negative']:<15} | {counts['false_positive']:<15} | {counts['false_negative']:<15} | {sumall:<10}\n")
        if self.timestamps_path:
            with open(self.timestamps_path, "w") as f:
                for stmp in self.timestamps:
                    f.write(f"{stmp}\n")


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
               plan = self.generate_plan()
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
               if clear_run:
                   print("[Agent] Clear run completed.")

        finally:
            self.end_time = time.time()
            self.brain._logger()
            self._logger(self.run_count-1)
            
               
    def execute_skill(self, skill_name, prev_skill):
        print(f"[Agent] Execute Skill")
        #-> read state
        prev_vector_state = self.read_state()
        #-> if there is not node for the skill, add one
        self.brain.create_node(skill_name)
        #-> Are preconditions of this skill met?
        preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state)
        #-> execute the skill
        skill = self.skillset[skill_name]
        success = skill.execute()
        vector_state = self.read_state()

        # collect stats &&
        #-> Is the outcome expectable? expectable = preconditions are met and success OR preconditions are not met and fail
        data = (prev_skill, prev_vector_state, skill_name, vector_state, success)
        if preconditions_are_met:
            if success:
                self.skill_stats[skill_name]['true_positive'] += 1
            else:
                self.skill_stats[skill_name]['false_negative'] += 1
                #-> Add unexpected data to buffor
                self.add_to_buffor(data)
        else:
            if success:
                self.skill_stats[skill_name]['false_positive'] += 1
                #-> Add unexpected data to buffor
                self.add_to_buffor(data)
            else:
                self.skill_stats[skill_name]['true_negative'] += 1

        if not skill_name == 'Finish':
            self.execute_skill('Finish', skill_name)
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

    def explore(self, prev_skill):
        print("[Agent] Explore")
        #-> choose random action (not checking the finish)
        available_skills = [k for k in self.skillset if k != "Finish"]
        if not available_skills:
            return None
        skill_name = random.choice(available_skills)
        #-> execute action
        self.execute_skill(skill_name, prev_skill)
        #-> return this action
        return skill_name
     
    def add_to_buffor(self, data):
        if len(self.fail_buffor) >= self.buffor_max_len:
            self.brain.update(self.fail_buffor)
            self.batch_update_count += 1
            self.fail_buffor = []
        self.fail_buffor.append(data)
        self.timestamps += [time.time()]


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
        if 'Finish' in nodes:
            # Check if there are any incoming connections to 'Finish'
            incoming_edges = self.brain.trans_graph.get_incoming_edges('Finish')
            if incoming_edges and len(incoming_edges) > 0:
                found_goal = True
                goal_skill = 'Finish'
                
        if not found_goal:
            goal_skill = random.choice(nodes)

        self.plan_count += 1
        # the paths
        plan_dir = os.path.join(self.pddl_dir, f"plan_{self.run_count}")

        os.makedirs(plan_dir, exist_ok=True)
        
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
            # Catches Pyperplan failing (e.g., syntax errors, unable to solve)
            print(f"[Agent] Pyperplan error: {e.stderr or e.stdout}")
            return []
        except Exception as e:
            # Catches literally any other unexpected error (e.g., OS errors, missing modules)
            print(f"[Agent] Unexpected error running Pyperplan: {e}")
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

                # Format is: (skill_name source_node target_node) 
                parts = line.strip("()").split()
                if len(parts) < 3:
                    continue
            
                action_name_lower = parts[0].lower()

                matched_skill = next(
                    (sk for sk in self.skillset.keys() if sk.lower() == action_name_lower), 
                    None
                )
                if matched_skill:
                    executable_plan.append(matched_skill) # <-- Just append the string
        if len(executable_plan)==0:
            print(f"[Agent] Error: no executable plan.")
        return executable_plan
