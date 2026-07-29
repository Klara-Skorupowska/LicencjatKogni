import cv2
import numpy as np
import subprocess
import os
import sys

import skill

class Brain():
    ''' Parent class for all the brains, i.e. logic of behaviour. Brain may use predefined skills in skillset'''
    def __init__(self, observations_map, actuators_map):
        self.observations_map = observations_map
        self.actuators_map = actuators_map
        self.skill_map = None # skills that the brain can use, to be set by the robot, as a map: skill name - skill object

    def step():
        ''' To be implemented by child classes. It should take observations and decide what to do. Returns data for actuators.'''
        raise NotImplementedError("Subclasses must implement this method")

class Ameba(Brain):
    ''' The simplest of the brains, it just moves forward and turns when it detects an obstacle. It does not use any skills.'''
    def __init__(self):
        self.actuators_map = {
            "left_motor": None,
            "right_motor": None
        }
        self.observations_map = {
            "dist_front": None,
            "dist_left": None,
            "dist_right": None
        }
        super().__init__(self.observations_map, self.actuators_map)

    def step(self, observations_map):
        self.observations_map = observations_map
        front, left, right = observations_map["dist_front"], observations_map["dist_left"], observations_map["dist_right"] 
        left_motor = 1.0
        right_motor = 1.0
        if front < 1: # obstacle ahead
            if left > right:
                left_motor = -0.5
                right_motor = 0.5
            else:
                left_motor = 0.5
                right_motor = -0.5
        self.actuators_map["left_motor"] = left_motor
        self.actuators_map["right_motor"] = right_motor
        return self.actuators_map

class AmebaPDDL(Brain):
    ''' 
        uses skillset and pddl to plan and execute a sequence of skills to achieve a goal. has defined lowlevel predicators, skills and results.
    '''
    def __init__(self):
        self.actuators_map = {
            "left_motor": None,
            "right_motor": None
        }
        self.observations_map = {
            "dist_front": None,
            "dist_left": None,
            "dist_right": None,
            "video": None
        }
        super().__init__(self.observations_map, self.actuators_map)
        
        self.plan = []           
        self.active_skill = None 
        self.skill_map = {
            "turn-until-clear": skill.Turn,
            "roam": skill.MoveForward
        }

    def check_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        min_px = (frame.shape[0] * frame.shape[1]) * 0.25

        mask_green = cv2.inRange(hsv, np.array([35, 100, 50]), np.array([85, 255, 255]))
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        if cv2.countNonZero(mask_green) > min_px: return "green"
        if cv2.countNonZero(mask_red) > min_px: return "red"
        return "none"

    def update_predicates(self, observations):
        color = self.check_color(observations["video"])
        if observations["dist_front"] < 1.0 and color != "green":
            return "(obstacle-ahead epuck2)"
        elif observations["dist_front"] < 1.0 and color == "green":
            return "(door-ahead epuck2)"
        else:
            return "(clear-path epuck2)"

    def generate_plan(self, initial_state_predicate):
        """ Writes problem.pddl and calls the Pyperplan solver """

        problem_str = f"""(define (problem find-door)
          (:domain epuck-arena)
          (:objects epuck2 - robot)
          (:init {initial_state_predicate})
          (:goal (door-ahead epuck2))
        )"""
        
        with open("problem.pddl", "w") as f:
            f.write(problem_str)
            
        print(f"Planning... Initial state: {initial_state_predicate}")

        # Call pyperplan to solve it
        result = subprocess.run(
            [sys.executable, "-m", "pyperplan", "domain.pddl", "problem.pddl"], 
            capture_output=True, text=True
        )
        
        # Parse the output plan
        plan = []
        if os.path.exists("problem.pddl.soln"):
            with open("problem.pddl.soln", "r") as f:
                for line in f:
                    # Strip the parentheses and robot name to get just the action name
                    action = line.strip().strip("()").split()[0]
                    plan.append(action)
            os.remove("problem.pddl.soln") # Cleanup
            
        print(f"Plan generated: {plan}")
        return plan

    def step(self, observations_map):

        # PLAN if no plan
        if not self.plan and self.active_skill is None:
            predicate = self.update_predicates(observations_map)
            self.plan = self.generate_plan(predicate)

        # SEQUENCE if have a plan and no active skill
        if self.plan and self.active_skill is None:
            next_action = self.plan.pop(0)
            self.active_skill = self.skill_map[next_action]()
            print(f"Executing skill: {next_action}")

        # ACT if have active skill
        if self.active_skill:
            actuators, is_done = self.active_skill.update(observations_map)
            if is_done:
                print(f"Skill {type(self.active_skill).__name__} completed.")
                self.active_skill = None
            self.actuators_map.update(actuators)
            return self.actuators_map

        return {"left_motor": 0.0, "right_motor": 0.0}