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
            'SpotTheDoor': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, min_dist, timeout=5.0),
            'GoToTheDoor': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, min_dist=min_dist, timeout=15.0),
            'GoThroughTheDoor': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, min_dist, timeout=15.0),
            'SpotTheGoal': SpotTheColor(120, self.camera, self.lidars, self.wheels, velocity, min_dist, timeout=5.0),
            'GoToTheGoal': GoToTheGoal(bus, 120, self.camera, self.lidars, self.wheels, velocity, min_dist=min_dist, timeout=15.0),
            'ClearThePath': ClearThePath(bus, self.camera, self.lidars, self.wheels, velocity, min_dist=min_dist, run_time=2.0, timeout=15.0),
        }
        
        # 1. Directory Structure Setup
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join("logs", timestamp)
        self.pddl_dir = os.path.join(self.log_dir, "PDDL")
        
        # 2. Add brain and pass it the root log directory
        self.brain = BrainNetwork(log_dir=self.log_dir)
        
        self.domain_path = os.path.join(self.pddl_dir, "domain.pddl")
        self.problem_path = os.path.join(self.pddl_dir, "problem.pddl")
        self.plan_file = os.path.join(self.pddl_dir, "problem.pddl.soln")

        # 3. Setup Skill Execution Logger
        self.skill_stats = {name: {'success': 0, 'fail': 0} for name in self.skillset.keys()}

    def run(self):
        try:
            n = 50
            for step in range(n):
                print(f"[AGENT] Step {step}/{n}")

                prev_state_id, prev_state_vector = self.read_state()

                if self.bus.call_service(f"/supervisor/ask/goal_zone"):
                    self.brain.transitional_map.nodes[prev_state_id]['is_goal'] = True

                # Look for a discovered true goal
                goal_node_id = None
                for node_id, node_data in self.brain.transitional_map.nodes(data=True):
                    if node_data.get('is_goal') in [True, 'True']: 
                        goal_node_id = node_id
                        break
                        
                is_temporary_goal = False
                if goal_node_id is None:
                    # Collect all existing graph nodes other than the current position
                    available_nodes = [
                        node for node in self.brain.transitional_map.nodes() 
                        if node != prev_state_id
                    ]
                    
                    if not available_nodes:
                        print("[AGENT] Graph empty or only current node known. Exploring...")
                        self.explore()
                        continue

                    # Select a random node as temporary target
                    goal_node_id = random.choice(available_nodes)
                    is_temporary_goal = True
                    print(f"[AGENT] Goal unknown. Temporary goal node: s{goal_node_id}")

                while True:
                    self.create_pddl(prev_state_id, goal_node_id, prev_state_vector)
                    plan = self.make_plan()
                    
                    if not plan:
                        print("[AGENT] No valid plan found.")
                        self.explore()
                        break 
                    
                    needs_explore = False
                    needs_replan = False
                    
                    for step_idx, (skill_name, target_state_str) in enumerate(plan):
                        skill = self.skillset[skill_name]
                        expected_state_id = int(target_state_str.replace('s', ''))
                        
                        success = skill.execute()
                        self.skill_stats[skill_name]['success' if success else 'fail'] += 1
                        
                        new_state_id, new_state_vector = self.read_state()
                        self.brain.update_gng(new_state_vector)
                        
                        if not success:
                            self.explore()
                            needs_explore = True
                            break 
                            
                        self.brain.update_tg(prev_state_vector, skill, new_state_vector)

                        # Check if the real goal zone was stumbled upon mid-plan
                        if self.bus.call_service(f"/supervisor/ask/goal_zone"):
                            self.brain.transitional_map.nodes[new_state_id]['is_goal'] = True
                            print("[AGENT] Discovered true goal zone during execution!")
                            if is_temporary_goal:
                                needs_replan = True
                                prev_state_id = new_state_id
                                prev_state_vector = new_state_vector
                                break
                        
                        if new_state_id != expected_state_id:
                            prev_state_id = new_state_id
                            prev_state_vector = new_state_vector
                            needs_replan = True
                            break 
                            
                        is_last_step = (step_idx == len(plan) - 1)
                        if not is_last_step:
                            prev_state_id = new_state_id
                            prev_state_vector = new_state_vector
                            continue
                            
                        # Reached the final step of the plan
                        if is_temporary_goal:
                            print(f"[AGENT] Reached temporary goal node s{goal_node_id}.")
                            prev_state_id = new_state_id
                            prev_state_vector = new_state_vector
                            break  # Exit the inner execution loop to select a new target or plan to true goal

                        if not self.bus.call_service(f"/supervisor/ask/goal_zone"):
                            prev_state_id = new_state_id
                            prev_state_vector = new_state_vector
                            needs_replan = True
                            break 
                        else:
                            print("[AGENT] Successful plan noted!")
                            success_path = os.path.join(self.pddl_dir, "successful_plan.txt")
                            with open(success_path, "w") as f:
                                f.write("Successful Plan Executed:\n")
                                for p_step, (p_skill, p_target) in enumerate(plan, 1):
                                    f.write(f"Step {p_step}: {p_skill} -> {p_target}\n")
                            return
                    
                    if needs_explore: break 
                    if needs_replan: continue 
                    if is_temporary_goal: break  # Re-evaluate in the outer loop
            print("[AGENT] The End")
                
        finally:
            if self.bus.call_service(f"/supervisor/ask/goal_zone"):
                print("[AGENT] Successfully reached goal")
                
            self.brain.log_graphs(stage='final')
            
            stats_path = os.path.join(self.log_dir, "skills_execution.txt")
            with open(stats_path, "w") as f:
                f.write(f"{'Skill Name':<25} | {'Successes':<10} | {'Fails':<10}\n")
                f.write("-" * 55 + "\n")
                for sk_name, counts in self.skill_stats.items():
                    f.write(f"{sk_name:<25} | {counts['success']:<10} | {counts['fail']:<10}\n")
               
    def read_state(self):
        print("[AGENT] Read State")
        self.lidars.read()
        self.camera.read()
        self.camera.preprocess()
        state_vector = np.concatenate([self.lidars.value, self.camera.coded]).flatten()
        self.brain._update_scaling(state_vector)
        state_id = self.brain.classify(state_vector)
        return state_id, state_vector

    def explore(self):
        print("[AGENT] Explore")
        while True:
            prev_state_id, prev_state_vector = self.read_state()
            if self.bus.call_service(f"/supervisor/ask/goal_zone"):
                self.brain.transitional_map.nodes[prev_state_id]['is_goal'] = True

            # Select skill using its name string so we can log it properly
            skill_name = random.choice(list(self.skillset.keys()))
            skill = self.skillset[skill_name]

            success = skill.execute()
            self.skill_stats[skill_name]['success' if success else 'fail'] += 1
            
            # Odczyt nowego stanu po KAŻDYM wykonaniu skilla
            new_state_id, new_state_vector = self.read_state() 
            
            # GNG uaktualniamy zawsze, niezależnie czy skill zakończył się sukcesem, czy błędem
            self.brain.update_gng(new_state_vector)
            
            if success:
                # Jeśli skill się powiódł, sprawdzamy czy przejście było już znane
                preddicted_state_id = self.brain.predict(prev_state_id, skill)
                
                if new_state_id != preddicted_state_id: 
                    # Uaktualniamy TG tylko przy sukcesie i gdy mapa przejść tego wymaga
                    self.brain.update_tg(prev_state_vector, skill, new_state_vector)
                
                # Przerywamy pętlę eksploracji po pierwszym udanym skillu
                return
            
    def create_pddl(self, start_node_id: int, goal_node_id: int, state_vector: np.ndarray):
        '''
        Creates the domain.pddl and problem.pddl strings on demand.
        '''
        print("[AGENT] Create PDDL")
        # Fetch the flat list of symbols directly
        symbols = self.brain.get_symbols()

        skills = set()
        states = set([start_node_id, goal_node_id]) 
        
        # Iterate over the list of symbols to gather skills and node states
        for sym in symbols:
            skills.add(sym['skill'])
            # Using .get for flexibility, defaulting to empty list if not found
            nodes = sym.get('raw_nodes', sym.get('nodes', [])) 
            for node in nodes:
                states.add(node)

        # Generate domain.pddl
        domain_str = "(define (domain robot-skills)\n"
        domain_str += "  (:requirements :typing)\n"
        domain_str += "  (:types state)\n"
        domain_str += "  (:predicates\n"
        for skill in skills:
            domain_str += f"    ({skill}_precondition ?s - state)\n"
            domain_str += f"    ({skill}_effect ?s - state)\n"
        domain_str += "    (robot-at ?s - state)\n  )\n\n"

        for skill in skills:
            domain_str += f"  (:action {skill}\n"
            domain_str += f"    :parameters (?from - state ?to - state)\n"
            domain_str += f"    :precondition (and (robot-at ?from) ({skill}_precondition ?from))\n"
            domain_str += f"    :effect (and (not (robot-at ?from)) (robot-at ?to) ({skill}_effect ?to))\n"
            domain_str += "  )\n"
        domain_str += ")"

        with open(self.domain_path, "w") as f:
            f.write(domain_str)

        # Generate problem.pddl
        problem_str = "(define (problem reach-end-pad)\n"
        problem_str += "  (:domain robot-skills)\n"
        problem_str += "  (:objects\n    "
        problem_str += " ".join([f"s{s}" for s in states]) + " - state\n  )\n"
        problem_str += "  (:init\n"
        problem_str += f"    (robot-at s{start_node_id})\n"
        
        # Add symbol predicates to the initial state
        for sym in symbols:
            symbol_name = sym.get('symbol_name', sym.get('name'))
            nodes = sym.get('raw_nodes', sym.get('nodes', []))
            for node in nodes:
                problem_str += f"    ({symbol_name} s{node})\n"
            
        problem_str += "  )\n"
        problem_str += "  (:goal\n"
        problem_str += f"    (robot-at s{goal_node_id})\n"
        problem_str += "  )\n)"

        with open(self.problem_path, "w") as f:
            f.write(problem_str)

    def make_plan(self):
        print("[AGENT] Make a Plan")
        if not os.path.exists(self.domain_path) or not os.path.exists(self.problem_path):
            return []

        if os.path.exists(self.plan_file):
            os.remove(self.plan_file)

        try:
            subprocess.run(
                [sys.executable, "-m", "pyperplan", self.domain_path, self.problem_path],
                capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError:
            return []

        if not os.path.exists(self.plan_file):
            return []

        executable_plan = []
        with open(self.plan_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"): continue 

                parts = line.strip("()").split()
                if len(parts) < 3: continue
            
                action_name_lower = parts[0]
                target_state = parts[2]

                matched_skill = next((sk for sk in self.skillset.keys() if sk.lower() == action_name_lower), None)
                if matched_skill:
                    executable_plan.append((matched_skill, target_state))

        return executable_plan