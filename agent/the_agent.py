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
        
        min_dist = 0.05
        velocity = 15
        self.skillset = {
            'SpotTheDoor': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, min_dist, timeout=3.0),
            'GoToTheDoor': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, min_dist=min_dist, timeout=15.0),
            'GoThroughTheDoor': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, min_dist, timeout=10.0),
            'SpotTheGoal': SpotTheColor(120, self.camera, self.lidars, self.wheels, velocity, min_dist, timeout=3.0),
            'GoToTheGoal': GoToTheGoal(bus, 120, self.camera, self.lidars, self.wheels, velocity, min_dist=min_dist, timeout=15.0),
            #'ClearThePath': ClearThePath(bus, self.camera, self.lidars, self.wheels, velocity, min_dist=min_dist, run_time=2.0, timeout=15.0),
        }
        
        # 1. Directory Structure Setup
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join("logs", timestamp)
        self.pddl_dir = os.path.join(self.log_dir, "PDDL")
        self.stats_path = os.path.join(self.log_dir, "execution_statistics.txt")
        
        # 2. Add brain and pass it the root log directory
        self.brain = BrainNetwork(log_dir=self.log_dir)
        
        self.domain_path = os.path.join(self.pddl_dir, "domain.pddl")
        self.problem_path = os.path.join(self.pddl_dir, "problem.pddl")
        self.plan_file = os.path.join(self.pddl_dir, "problem.pddl.soln")

        # 3. Setup Skill Execution Logger
        self.skill_stats = {name: {'success': 0, 'fail': 0} for name in self.skillset.keys()}
        self.skill_stats['GoalReached'] =  {'success': 0, 'fail': 0}
        self.start_time = None
        self.end_time = None
        self.run_count = 0

    def _logger(self, run, iteration):
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
            f.write(f"run: {run}, iteration: {iteration}\n")
            f.write(f"{'Skill Name':<25} | {'Successes':<10} | {'Fails':<10}\n")
            f.write("-" * 55 + "\n")
            for sk_name, counts in self.skill_stats.items():
                if not sk_name == 'GoalReached': 
                    sumall = counts['success'] + counts['fail']
                    f.write(f"{sk_name:<25} | {counts['success']:<10} | {counts['fail']:<10} | {sumall:<10}\n")
                else:
                    counts['fail'] = self.run_count - counts['success']
                    f.write(f"{sk_name:<25} | {counts['success']:<10} | {counts['fail']:<10} | {self.run_count:<10} // one per run\n") # math not mathing

        pass
    def run(self):
        try:
            self.start_time = time.time()
            N = 10
            n = 15
            clear_run = False # clear run = to the true goal, without exploring
            while not clear_run:
                self.run_count +=1
                clear_run = True
                #> read state
                prev_state_id, prev_state_vector = self.read_state()

                for step in range(n):
                    print(f"[Agent] Step {step}/{n} in run {self.run_count} ------------------------- ")
                    self._logger(self.run_count, step)
                    #> read state = prev state from previous step
                    #> create PDDL
                     # you need goal node:
                    # 1. Resolve current state to its active transitional map identity
                    current_tg_node = str(self.brain._get_merged_node(prev_state_id))

                    # 2. Check for the true goal, ensuring the agent is not already standing on it
                    goal_node_id = None
                    for node_id, node_data in self.brain.transitional_map.nodes(data=True):
                        node_str = str(node_id)
                        if node_data.get('is_goal') in [True, 'True']:
                            if node_str != current_tg_node:
                                goal_node_id = node_id
                                print(f"[Agent] Creating plan for the true goal ({goal_node_id}).")
                                break
                            else:
                                print(f"[Agent] Agent is already at the true goal node ({node_str}).")

                    # 3. Fallback: Select an exploratory goal strictly distinct from current node
                    if goal_node_id is None:
                        active_nodes = [node_id for node_id in self.brain.transitional_map.nodes()]
                        if active_nodes:
                            goal_node_id = random.choice(active_nodes)
                            if goal_node_id == current_tg_node:
                                print(f"[Agent] Current node drawn as goal ({goal_node_id}).")
                                clear_run = False
                                self.explore()
                                prev_state_id, prev_state_vector = self.read_state()
                                continue
                            print(f"[Agent] Creating plan for random goal ({goal_node_id}).")
                            clear_run = False
                        else: 
                            print(f"[Agent] No active nodes other than current state ({current_tg_node}).")
                            clear_run = False
                            self.explore()
                            prev_state_id, prev_state_vector = self.read_state()
                            continue

                    self.create_pddl(prev_state_id, goal_node_id)
                
                    #> make a plan
                    plan = self.make_plan()
                    curr_state_id = prev_state_id; curr_state_vector = prev_state_vector
                    #> if no plan, explore
                    if not plan:
                        print("[Agent] Plan is empty")
                        clear_run = False
                        self.explore()
                        curr_state_id, curr_state_vector = self.read_state()
                    else:
                        for step_idx, (skill_name, target_state_str) in enumerate(plan):
                            #> execute next step
                            success, skill = self.execute_skill(skill_name)
                            curr_state_id, curr_state_vector = self.read_state()
                            if not success: 
                                clear_run = False
                                self.explore()
                                break  
                            resolved_preds = self.brain.resolve_predicates(curr_state_vector)
                            expected_pred = ("at", f"node_{target_state_str}")
                            if expected_pred not in resolved_preds:
                                print(f"[Agent] Execution divergence: expected {expected_pred}, got {resolved_preds}")
                                self.brain.update_gng(curr_state_vector)
                                self.brain.update_tg(prev_state_vector, skill_name, curr_state_vector)
                                break
                    prev_state_id = curr_state_id
                    prev_state_vector = curr_state_vector
                    if self.bus.call_service(f"/supervisor/ask/goal_zone"):
                        print ("[Agent] Goal reached. Restart.")
                        self.bus.call_service(f"/supervisor/do/restart")
                if not clear_run:
                    print ("[Agent] Run ended. Restart.")
                    self.bus.call_service(f"/supervisor/do/restart")
                else:
                    print ("[Agent] Run ended. Successful plan execution.")
                if self.run_count >= N:
                    print(f"[Agent] I give up. IT IS IMPOSSIBLE!!!")
                    break
        finally:
            print("[Agent] Run completed.")
            self.end_time = time.time()
            self.brain._log_graphs()
            self._logger(N, n)
            
               
    def execute_skill(self, skill_name):
        skill = self.skillset[skill_name]
        success = skill.execute()
        _, state_vector = self.read_state()
        self.skill_stats[skill_name]['success' if success else 'fail'] += 1 # collect skill execution stats
        self.brain.update_gng(state_vector) # always update GNG with the new state after skill execution
        return success, skill

    def read_state(self):
        print("[Agent] Read State")
        self.lidars.read()
        self.camera.read()
        self.camera.preprocess()
        state_vector = np.concatenate([self.lidars.value, self.camera.coded]).flatten()

        # Resolve ground truth PDDL propositions from continuous state
        active_predicates = self.brain.resolve_predicates(state_vector)
        
        # Extract the current location object ('node_X')
        current_loc = next((args[0] for pred, *args in active_predicates if pred == "at"), None)
        state_id = current_loc.replace("node_", "") if current_loc else str(self.brain.classify(state_vector))

        if self.bus.call_service(f"/supervisor/ask/goal_zone"):
            merged_id = self.brain._get_merged_node(state_id)
            if not self.brain.transitional_map.has_node(merged_id):
                self.brain.transitional_map.add_node(merged_id, aliases=[state_id])
            self.brain.transitional_map.nodes[merged_id]['is_goal'] = True
            self.skill_stats['GoalReached']['success'] += 1

        return state_id, state_vector

    def explore(self):
        print("[Agent] Explore")
        while True:
            #> read state
            prev_state_id, prev_state_vector = self.read_state()
            #> select a random skill
            skill_name = random.choice(list(self.skillset.keys()))
            success, skill = self.execute_skill(skill_name)
            if success:
                break  
        #> read state
        _, curr_state_vector = self.read_state()
        #> predictable?
        resolved_preds = self.brain.resolve_predicates(curr_state_vector)
        expected_pred = self.brain.predict(prev_state_vector, skill_name)
        if expected_pred not in resolved_preds:
            self.brain.update_gng(curr_state_vector)
            self.brain.update_tg(prev_state_vector, skill_name, curr_state_vector)

    def create_pddl(self, start_node_id: str, goal_node_id: str):
        print("[Agent] Create PDDL")
        print(f"[Agent] Start (node): {start_node_id} | Goal (node): {goal_node_id}")
        self.brain.create_symbols()

        domain_str = self.brain.generate_domain_pddl()
        with open(self.domain_path, "w") as f:
            f.write(domain_str)

        problem_str = self.brain.generate_problem_pddl(initial_state=start_node_id, goal_state=goal_node_id)
        with open(self.problem_path, "w") as f:
            f.write(problem_str)


    def make_plan(self):
        print("[Agent] Make a Plan")
        if not os.path.exists(self.domain_path) or not os.path.exists(self.problem_path):
            print(f"[Agent] Error: no .pddl files")
            return []

        if os.path.exists(self.plan_file):
            os.remove(self.plan_file)

        try:
            res = subprocess.run(
                [sys.executable, "-m", "pyperplan", self.domain_path, self.problem_path],
                capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            # Print compiler errors to catch missing objects or domain syntax issues
            print(f"[Agent] Pyperplan error: {e.stderr or e.stdout}")
            return []

        if not os.path.exists(self.plan_file):
            return []

        executable_plan = []
        with open(self.plan_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue 

                # Format is: (skill_name source_node target_node)
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