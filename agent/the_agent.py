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
    # helper struct
    class ExecutedSkill():
        def __init__(self, skill_name: str, succeed: bool):
            self.name = skill_name
            self.succeed = succeed

    def __init__(self, bus: Communicator, how_many_runs: int, how_many_tests: int, source_dir = 'new'):
        super().__init__(bus)
        self.wheels = WheelsActuator(self.bus)
        self.lidars = VirtualSensorArray(self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]])
        self.camera = CameraSensor(self.bus)
        
        self.max_runs = how_many_runs       # how many explore actions
        self.N_tests = how_many_tests       # how many starts from different positions with ready predicates

        self.fail_buffor = []
        self.buffor_max_len = 10 

        self.current_skill = None
        self.bus.register_service('agent/ask/action', self._give_current_action)

        save_dist = 0.05
        velocity = 15
        self.skillset = {
            'Start':Start(bus,self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=1.0),
            'Find_the_Door': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=4.0),
            'Aproach_the_Door': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'Open_the_Door': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=10.0),
            'Find_the_Goal': SpotTheColor(26, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=4.0),
            'Aproach_the_Goal': GoToTheGoal(bus, 26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'Finish': Finish(bus, 26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=1.0)
        }
        
        # 1. Directory Structure Setup
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join("logs", timestamp)
        self.continued_from = None
        resolved_source_dir = None
        
        if source_dir == 'new':
            pass
        elif source_dir == 'continue':
            resolved_source_dir = self._get_latest_session("logs")
            if resolved_source_dir:
                self.continued_from = resolved_source_dir
            else:
                print(f"[Agent] No previous session found. Creating new one.")
        else:
            resolved_source_dir = os.path.join("logs", source_dir)
            if os.path.isdir(resolved_source_dir):
                self.continued_from = resolved_source_dir
            else:
                print(f"[Agent] Directory {resolved_source_dir} does not exist. Creating new one.")
                resolved_source_dir = None

        self.pddl_dir = os.path.join(self.log_dir, "PDDL")
        self.stats_path = os.path.join(self.log_dir, "execution_statistics.txt")
        self.timestamps_path = os.path.join(self.log_dir, "timestamps.txt")
        
        # 2. Add brain and pass it the root log directory
        self.brain = BrainNetwork(log_dir=self.log_dir)
        self.brain.load_brain(resolved_source_dir)

        # 3. Setup Execution Logger
        self.skill_stats = {name: {'true_positive': 0, 'true_negative': 0, 'false_positive': 0, 'false_negative': 0} for name in self.skillset.keys()}
        self.logs = []
        self.tests = []
        self.start_time = None
        self.end_time = None
        self.run_count = 0
        self.batch_update_count = 0
        self.timestamps = []

    def _give_current_action(self, request=None):
        return self.current_skill

    def _get_latest_session(self, logs_root):
        target_dir = None
        # Find the latest under logs_root
        if os.path.exists(logs_root):
            session_dirs = [
                os.path.join(logs_root, d) for d in os.listdir(logs_root) 
                if os.path.isdir(os.path.join(logs_root, d))
            ]
            if session_dirs:
                target_dir = max(session_dirs, key=os.path.basename)

        if not target_dir:
            return None
        return target_dir

    def _logger(self, run, tests_runned = False):
        if self.stats_path:
            if self.end_time is None: 
                end_time = time.time()
            else:
                end_time = self.end_time
            with open(self.stats_path, "w") as f:
                # -> Inject continuation info if it exists
                if self.continued_from:
                    f.write(f"CONTINUATION FROM SESSION: {self.continued_from}\n")
                    f.write("-" * 100 + "\n")

                start_str = datetime.fromtimestamp(self.start_time).strftime("%d-%m-%Y %H:%M:%S")
                end_str = datetime.fromtimestamp(end_time).strftime("%d-%m-%Y %H:%M:%S")
                
                duration = end_time - self.start_time
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
                f.write("-" * 100 + "\n")
                for log in self.logs:
                    f.write(f"{log}/n")
                if tests_runned:
                    f.write("-" * 100 + "\n")
                    f.write(f"{'Test':<20} | {'Plan Length':<15} | {'Steps Executed':<15} | {'Last Action':<15} | {'Duration [s]':<15} | {'Result':<15}\n")
                    successes = 0
                    for test in self.tests:
                        run, length, steps_executed, last_step, duration, result = test
                        f.write(f"{run+1:<20} | {length:<15} | {steps_executed:<15} | {last_step:<15} | {duration:<15.3f} | {result:<10}\n")
                        if result:
                            successes += 1
                    f.write(f"tests: {self.N_tests}, successes: {successes}\n")

        if self.timestamps_path:
            with open(self.timestamps_path, "w") as f:
                for stmp in self.timestamps:
                    f.write(f"{stmp}\n")


    def run(self):
        try:
           self.run_count = 0
           self.start_time = time.time()
           prev_skill = self.ExecutedSkill(None, None)

           while self.run_count < self.max_runs:
               self._logger(self.run_count)
               self.run_count += 1
               print(f"[Agent] Run {self.run_count}/{self.max_runs} ---------------")
               #-> Generate Plan
               plan, start_skill, finish_skill = self.generate_plan()
               if len(plan) < 1: # what if plan is empty? Explore
                   print("[Agent] Returned empty plan.")
                   prev_skill = self.explore(prev_skill) 
                   continue # the while loop
               #-> Execute Next Step of a Plan
               print("[Agent] Start executing plan.")
               plan_completed = True
               for i, skill_name in enumerate(plan):
                   print(f"[Agent] Executing step {i+1}/{len(plan)}: {skill_name}")
                   success = self.execute_skill(skill_name, prev_skill)
                   #-> Was it successful?
                   if not success: #-> if not, explore
                       print("[Agent] Failed action. Abandoning the plan.")
                       prev_skill = self.explore(prev_skill)
                       plan_completed = False
                       break # the for loop
                   #-> if yes, do the next step of the plan
                   prev_skill.name = skill_name
                   prev_skill.succeed = True
               if start_skill == 'Start' and finish_skill == 'Finish' and plan_completed:
                   print("[Agent] Clear Run from Start to Finish.")
                   break

        finally:
            self.end_time = time.time()
            self.brain._logger()
            self.test_run()
            self._logger(self.run_count, tests_runned=True)
            
               
    def execute_skill(self, skill_name, prev_skill):
        print(f"[Agent] Execute Skill")
        self.current_skill = skill_name
        #-> read state
        prev_vector_state = self.read_state()
        #-> if there is not node for the skill, add one
        self.brain.create_node(skill_name)
        #-> Are preconditions of this skill met?
        preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state)
        #-> execute the skill
        skill = self.skillset[skill_name]
        succeed = skill.execute()
        vector_state = self.read_state()

        # collect stats &&
        #-> Is the outcome expectable? expectable = preconditions are met and success OR preconditions are not met and fail
        data = (prev_skill.name, prev_vector_state, prev_skill.succeed, skill_name, vector_state, succeed)
        if preconditions_are_met:
            if succeed:
                self.skill_stats[skill_name]['true_positive'] += 1
            else:
                self.skill_stats[skill_name]['false_negative'] += 1
                #-> Add unexpected data to buffor
                self.add_to_buffor(data)
        else:
            if succeed:
                self.skill_stats[skill_name]['false_positive'] += 1
                #-> Add unexpected data to buffor
                self.add_to_buffor(data)
            else:
                self.skill_stats[skill_name]['true_negative'] += 1

        if not skill_name == 'Finish':
            self.execute_skill('Finish', self.ExecutedSkill(skill_name, succeed))
        return succeed

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
        succeed = self.execute_skill(skill_name, prev_skill)
        #-> return this action
        return self.ExecutedSkill(skill_name, succeed)
     
    def add_to_buffor(self, data):
        if len(self.fail_buffor) >= self.buffor_max_len:
            self.brain.update(self.fail_buffor)
            self.batch_update_count += 1
            self.fail_buffor = []
        self.fail_buffor.append(data)
        self.timestamps += [time.time()]


    def generate_plan(self, start_skill = None, goal_skill = None):
        print("[Agent] Generate Plan")
        #-> read state
        vector_state = self.read_state()
        #-> get active skills: by resolving predicates, which skills are doable right now
        active_skills = self.brain.get_active_skills(vector_state)
        if not active_skills: # if no active skills (nodes), no plan at all
            print(f"[Agent] No active skills.")
            return [], None, None 
        #-> choose the starting skill from active ones
        if start_skill is None:
            start_skill = random.choice(active_skills)
        #-> choose the goal from nodes in the transgraph
        if goal_skill is None:
            nodes = self.brain.get_nodes()
            nodes =  [n for n in nodes if n != start_skill]
            if not nodes:
                print(f"[Agent] No nodes in transitional graph.")
                return [], None, None
            goal_skill = random.choice(nodes)

        print(f"[Agent] Generating plan from {start_skill} to {goal_skill}.")

        # the paths
        plan_dir = os.path.join(self.pddl_dir, f"newest")
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
            return [], None, None

        if os.path.exists(plan_file):
            os.remove(plan_file)

        try:
            res = subprocess.run(
                [sys.executable, "-m", "pyperplan", "-s", "astar", "-H", "hff", domain_file, problem_file],
                capture_output=True, text=True, check=True,
                timeout=10.0  # Prevents hanging when no plan exists
            )
        except subprocess.TimeoutExpired:
            print(f"[Agent] Pyperplan timed out (no plan reachable).")
            self.logs += [f"[Pyperplan] [RUN {self.run_count}]  Pyperplan timed out."]
            return [], None, None
        except subprocess.CalledProcessError as e:
            # Catches Pyperplan failing (e.g., syntax errors, unable to solve)
            print(f"[Agent] Pyperplan error: {e.stderr or e.stdout}")
            return [], None, None
        except Exception as e:
            # Catches literally any other unexpected error (e.g., OS errors, missing modules)
            print(f"[Agent] Unexpected error running Pyperplan: {e}")
            return [], None, None

        if not os.path.exists(plan_file):
            return [], None, None

        #-> change to executable plan
        executable_plan = []
        with open(plan_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue 

                parts = line.strip("()").split()
                if len(parts) < 1:
                    continue
            
                action_name_lower = parts[0].lower()

                matched_skill = next(
                    (sk for sk in self.skillset.keys() if sk.lower() == action_name_lower), 
                    None
                )
                if matched_skill:
                    executable_plan.append(matched_skill) 
        if len(executable_plan)<=0:
            print(f"[Agent] Error: no executable plan.")
        else:
            executable_plan.append(goal_skill)
            # move executable plan to not temporary folder
            '''
            exec_plan_dir = os.path.join(self.pddl_dir, f"plan_{self.run_count}")
            os.rename(plan_dir, exec_plan_dir)
            '''
        return executable_plan, start_skill, goal_skill

    def  test_run(self):
        for i in range(self.N_tests):
            print(f"[Agent] Test {i}/{self.N_tests}.")
            start = time.time()
            # restart
            self.bus.call_service(f"/supervisor/do/restart")
            # run pddl solver
            plan, _, _ = self.generate_plan('Start', 'Finish')
            # execute plan
            if len(plan) < 1:
                steps_executed = 0
                last_step = '-'
                result = False
            else:
                result = True
                print(f"[Agent] Executing plan: {plan}")
                for step, skill_name in enumerate(plan):
                    print(f"[Agent] Executing step {step}/{len(plan)}: {skill_name}")
                    prev_vector_state = self.read_state()
                    preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state)
                    if not preconditions_are_met:
                        print(f"[Agent] Preconditions are not met: {skill_name}")
                        steps_executed = step
                        result = False
                        break
                    skill = self.skillset[skill_name]
                    succeed = skill.execute()
                    last_step = skill_name
                    steps_executed = step + 1
                    '''
                    # check from supervisor
                    if not succeed:
                        result = False
                        break
                    '''
            duration = time.time() - start 
            self.tests.append((i, len(plan), steps_executed, last_step, duration, result))
            if result:
                print(f"[Agent] Test succeed.")
            else:
                print(f"[Agent] Test failed.")

