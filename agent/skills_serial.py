import random
from turtle import clear

from .skills import *
from communicator import Communicator

import time
import cv2
import numpy as np

### all skills required to achive the goal, in order ###
# if sensors are not enough, the supervisor decides if the skill is succesfull
# 1) spot the door
# 2) go to the door
# 3) go through the door
# 4) spot the finish pad
# 5) go to the finish pad

class SpotTheColor(Skill):
    """
    Turns around until it sees the <color> object, centers it,
    and returns True. Fails if an obstacle is too close or timeout is reached.
    """

    def __init__(
        self,
        hue: int,
        camera: CameraSensor,
        lidars: VirtualSensorArray,
        wheels: WheelsActuator,
        velocity: float,
        min_dist: float,
        timeout: float = 10.0,
    ):
        super().__init__()
        assert 0 <= hue <= 179, f"Hue {hue} out of bounds (0-179)"

        self.hue = hue
        self.camera = camera
        self.lidars = lidars
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.timeout = timeout

    def execute(self) -> bool:
        class_name = self.__class__.__name__
        print(f"[Agent] Executing {class_name} skill. Hue: {self.hue}.")

        hue_tol = 25

        lower_HSV = np.array([max(0, self.hue - hue_tol), 50, 50])
        upper_HSV = np.array([min(179, self.hue + hue_tol), 255, 255])
        center_tolerance = 15  # Pixels within horizontal center to consider aligned

        start_time = time.time()
        while time.time() - start_time < self.timeout:

            # 1. Camera Snapshot & HSV Detection
            frame = self.camera.read()
            if frame is None:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_HSV, upper_HSV)

            # 2. Locate Green Centroid
            moments = cv2.moments(mask)
            frame_center_x = frame.shape[1] // 2

            if moments["m00"] > 100:  # Green detected (filtering noise)
                cx = int(moments["m10"] / moments["m00"])

                # Check if door is centered
                if abs(cx - frame_center_x) <= center_tolerance:
                    self.wheels.set_parameters([0.0, 0.0])
                    print(f"[Agent] {self.__class__.__name__} finished successfully.")
                    return True
                elif cx < frame_center_x:
                    # Door is to the left -> Turn Left
                    self.wheels.set_parameters([-self.velocity, self.velocity])
                else:
                    # Door is to the right -> Turn Right
                    self.wheels.set_parameters([self.velocity, -self.velocity])
            else:
                # No green found: continue searching by rotating
                self.wheels.set_parameters([self.velocity, -self.velocity])

            time.sleep(0.01)

        # Timeout reached
        self.wheels.set_parameters([0.0, 0.0])
        print(f"[Agent] {class_name} timed out.")
        return False

class GoToTheDoor(Skill):
    '''
    Approaches the door keeping it centered. Once at target_distance, 
    it performs side-step maneuvers until perfectly centered in front of the green door.
    '''
    def __init__(
        self,
        camera: CameraSensor, 
        lidars: VirtualSensorArray, 
        wheels: VirtualActuator, 
        velocity: float, 
        min_dist: float, 
        target_distance: float = 0.05,
        distance_tolerance: float = 0.005,
        timeout: float = 15.0
    ):
        super().__init__()
        self.camera = camera 
        self.lidars = lidars
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.timeout = timeout
        self.start_time = None
        self.center_tolerance = 20

    def _get_green_cx(self, frame):
        """Returns the X centroid of the green object, or None if not found."""
        if frame is None: return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        moments = cv2.moments(mask)
        if moments["m00"] > 100:
            return int(moments["m10"] / moments["m00"])
        return None

    def _turn_to_the_wall(self, direction) -> bool:
        self.wheels.set_parameters([0.0, 0.0])
        distances = self.lidars.read()
        # turning
        while distances[0] == 0.1 or distances[7] == 0.1 or abs(distances[0] - distances[7]) > self.distance_tolerance:
            if (time.time() - self.start_time) > self.timeout: return False
            distances = self.lidars.read()
            if distances[0] > distances[7]:
                self.wheels.set_parameters([self.velocity, 0])
            elif distances[0] < distances[7]:
                self.wheels.set_parameters([0, self.velocity])
            else:
                if direction == 'LEFT':
                    self.wheels.set_parameters([0, self.velocity])
                else:
                    self.wheels.set_parameters([self.velocity, 0])
            time.sleep(0.01)
        self.wheels.set_parameters([0.0, 0.0])
        # distance to target_distance
        while True:
            distances = self.lidars.read()
            distance = (distances[0] + distances[7])/2
            if abs(distance - self.target_distance) < self.distance_tolerance: break
            elif distance > self.target_distance:
                v = self.velocity
            else:
                v = - self.velocity
            self.wheels.set_parameters([v, v])
            time.sleep(0.01)
            if time.time() - self.start_time > self.timeout: return False
        return True 

    def _turn_from_the_wall(self, direction) -> bool:
        # turn away 90 deg, said dir
        self.wheels.set_parameters([0, 0])
        distances = self.lidars.read()
        v_left = self.velocity if direction == 'RIGHT' else -self.velocity
        v_right = - v_left
        side_idx = 5 if direction == 'LEFT' else 2
        clear_path = ( distances[0] > self.min_dist and distances[7] > self.min_dist )
        minimal_dist = float(np.min(distances))
        parallel = ( abs(minimal_dist - distances[side_idx]) < self.distance_tolerance ) # closest point is the side
        side = (distances[side_idx-1] > distances[side_idx+1]) # change in this means that we passed the parallel point
        # turn
        while not clear_path or not parallel: 
            distances = self.lidars.read()
            clear_path = ( float(np.min([distances[0:1], distances[6:7]])) > self.min_dist ) 
            parallel = ( np.min([distances[side_idx-1], distances[side_idx+1]]) > distances[side_idx] )
            if not clear_path:
                self.wheels.set_parameters([-self.velocity, -self.velocity])
                continue
            self.wheels.set_parameters([v_left, v_right])
            if time.time() - self.start_time > self.timeout: return False
            if not (side == (distances[side_idx-1] > distances[side_idx+1])): 
                break
            time.sleep(0.01)
        return True 

    def _align(self) -> bool:
        time_step = 1.0
        frame = self.camera.read()
        cx = self._get_green_cx(frame)
        if cx is None: return False
        frame_center_x = frame.shape[1] // 2
        dir_new = 'RIGHT' if cx > frame_center_x else 'LEFT'
        dir_old = None
        while time.time() - self.start_time < self.timeout:
            distances = self.lidars.read()
            frame = self.camera.read()
            cx = self._get_green_cx(frame)
            if cx is None: return False
            frame_center_x = frame.shape[1] // 2
            if abs(frame_center_x - cx) < self.center_tolerance: return True # centered
            # the aligning using binary search. 
            # [START] turn 90 deg from wall in the dir of cx, 
            dir_old = dir_new
            dir_new = 'RIGHT' if cx > frame_center_x else 'LEFT'
                            # if direction changed: decrease <time_step>
            if not (dir_new == dir_old): time_step = time_step/2
            if not self._turn_from_the_wall(dir_new):
                print("[DEBUG] ---------------------------------- FROM TURN") 
                return False
            self.wheels.set_parameters([0, 0])
            # move following the wall/door for <time_step>, 
            step_start = time.time()
            while time.time() - step_start < time_step: # forward walk
                self.wheels.set_parameters([self.velocity, self.velocity])
                distances = self.lidars.read()
                if float(np.min([distances[0:1], distances[6:7]])) > self.min_dist and not self._turn_from_the_wall(dir_new):
                    print("[DEBUG] ---------------------------------- za blisko ściany")
                    print("[DEBUG] ---------------------------------- FROM TURN") 
                    return False
                if time.time() - self.start_time > self.timeout: return False
                time.sleep(0.01)
            self.wheels.set_parameters([0, 0])
            # turn back, 
            dir_back = 'RIGHT' if dir_new == 'LEFT' else 'LEFT'
            if not self._turn_to_the_wall(dir_back): 
                print("[DEBUG] ---------------------------------- TO TURN") 
                return False
            self.wheels.set_parameters([0, 0])
            # check cx: if centered return true, if not go back to step [START]
            # return False after timed out
        return False

    def execute(self) -> bool:
        print(f"[Agent] Executing {self.__class__.__name__} skill")
        self.start_time = time.time()
        while time.time() - self.start_time < self.timeout:        
            # snapshot: cx and lidars
            frame = self.camera.read()
            distances = self.lidars.read()
            cx = self._get_green_cx(frame)
            # if clear path (front lidars > max(min_dist, target_distance): approach door, correcting path using cx
                # if cx = None -> return False (lost track of the door)
            if cx is None:
                self.wheels.set_parameters([0.0, 0.0])
                print(f"[Agent] {self.__class__.__name__} failed. No door in sight.")
                return False

            min_front = float(np.min([distances[0:2], distances[5:7]]))
            if abs(min_front - np.max([self.min_dist, self.target_distance])) > self.distance_tolerance:
                err = (cx - frame.shape[1] // 2)/frame.shape[1]
                steer = err * self.velocity
                v_left = self.velocity + steer
                v_right = self.velocity - steer
                self.wheels.set_parameters([v_left, v_right])
                clear_path = ( float(np.min([distances[0:1], distances[6:7]])) > self.min_dist )
                if not clear_path: 
                    self.wheels.set_parameters([0, 0])
                    # are we here?
                    if self._align():
                        print(f"[Agent] {self.__class__.__name__} finished successfully.")
                        return True
                    # or not
                    print(f"[Agent] {self.__class__.__name__} failed. Obstacle ahead.")
                    self.wheels.set_parameters([0, 0])
                    return False
            else:
            # if not: turn to the wall/door so both front lidars have the same value 
                direction = 'RIGHT' if distances[0] > distances[7] else 'LEFT'
                if not self._turn_to_the_wall(direction): 
                    print(f"[Agent]  {self.__class__.__name__} failed. Error while turning to the wall")
                    return False
                aligned = self._align()
                # if align == True, align the distance to target_distance -> return True
                # if false -> return False
                if aligned:
                    print(f"[Agent] {self.__class__.__name__} finished successfully.")
                    return True
                else:
                    print(f"[Agent] {self.__class__.__name__} failed.")
                    return False
            # if timeout -> return False
        # Timeout reached
        self.wheels.set_parameters([0, 0])
        print("[Agent] Timed out.")
        return False

class GoThroughTheDoor(Skill):
    '''
    Opens door, any color.
    '''
    def __init__(
        self,
        camera: CameraSensor, 
        lidars: VirtualSensorArray, 
        wheels: VirtualActuator, 
        velocity: float, 
        min_dist: float,
        timeout: float = 15.0
    ):
        super().__init__()
        self.camera = camera 
        self.lidars = lidars
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.timeout = timeout
        self.start_time = None

    def execute(self)-> bool:
        print(f"[Agent] Executing {self.__class__.__name__} skill")
        # Start moving forward
        self.wheels.set_parameters([self.velocity, self.velocity])
        distances = self.lidars.read()
        front_left_dist = distances[0]
        front_right_dist = distances[7]

        if front_left_dist > self.min_dist and front_right_dist > self.min_dist:
            print(f"[Agent] {self.__class__.__name__} failed. Not in front of the door.")
            return False  # No door detected in front, cannot open

        start_time = time.time()
        while time.time() - start_time < self.timeout:
            # 1. Take a simultaneous snapshot of all required sensors
            distances = self.lidars.read()
            left_dist = distances[2]
            right_dist = distances[5]
            front_left_dist = distances[0]
            front_right_dist = distances[7]
            
            # 2. Evaluate the snapshot
            door_on_left = left_dist < self.min_dist
            door_on_right = right_dist < self.min_dist
            front_is_clear = (front_left_dist > self.min_dist) and (front_right_dist > self.min_dist)
            if (door_on_left or door_on_right) and front_is_clear:
                self.wheels.set_parameters([0.0, 0.0]) # halt
                print(f"[Agent] {self.__class__.__name__} finished successfully.")
                return True # there was a door and we opened it successfully
            
            # 3. Yield to the CPU
            time.sleep(0.01)
            
        # Timeout reached = there was a door but we failed to open it in time
        self.wheels.set_parameters([0.0, 0.0])
        print(f"[Agent] {self.__class__.__name__} failed. Timed out.")
        return False 

class GoToTheGoal(Skill):
    '''
    Approaches the goal keeping it centered. Once at target_distance, it stops.
    '''
    def __init__(
        self,
        camera: CameraSensor, 
        lidars: VirtualSensorArray, 
        wheels: VirtualActuator, 
        velocity: float, 
        min_dist: float, 
        target_distance: float = 0.05,
        distance_tolerance: float = 0.005,
        timeout: float = 20.0
    ):
        super().__init__()
        self.camera = camera 
        self.lidars = lidars
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.timeout = timeout
        self.start_time = None
        self.center_tolerance = 20

    def _get_blue_cx(self, frame):
        """Returns the X centroid of the blue object, or None if not found."""
        if frame is None: return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([95, 50, 50]), np.array([135, 255, 255]))
        moments = cv2.moments(mask)
        if moments["m00"] > 100:
            return int(moments["m10"] / moments["m00"])
        return None

    def execute(self):
        print(f"[Agent] Executing {self.__class__.__name__} skill")
        self.start_time = time.time()
        while time.time() - self.start_time < self.timeout:        
            # snapshot: cx and lidars
            frame = self.camera.read()
            distances = self.lidars.read()
            cx = self._get_blue_cx(frame)
            # if clear path (front lidars > max(min_dist, target_distance): approach door, correcting path using cx
                # if cx = None -> return False (lost track of the door)
            if cx is None:
                self.wheels.set_parameters([0.0, 0.0])
                print(f"[Agent] {self.__class__.__name__} failed. No goal in sight.")
                return False

            min_front = float(np.min([distances[0:2], distances[5:7]]))
            centered = abs(cx - frame.shape[1] // 2) < self.center_tolerance
            if abs(min_front - self.target_distance) < self.distance_tolerance:
                if centered:
                    self.wheels.set_parameters([0, 0])
                    print(f"[Agent] {self.__class__.__name__} finished successfully.")
                    return True
            clear_path = ( float(np.min([distances[0:1], distances[6:7]])) > self.min_dist )
            if not clear_path and not centered:
                self.wheels.set_parameters([0, 0])
                print(f"[Agent] {self.__class__.__name__} failed. Obstacle ahead.")
                return False

            err = (cx - frame.shape[1] // 2) / frame.shape[1]
            steer = err * self.velocity * 2
            v_left = self.velocity + steer
            v_right = self.velocity - steer
            self.wheels.set_parameters([v_left, v_right])

        # Timeout reached
        self.wheels.set_parameters([0, 0])
        print(f"[Agent] {self.__class__.__name__} failed. Timed out.")
        return False
