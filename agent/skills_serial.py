from turtle import goto

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
        print(f"[AGENT] Executing {class_name} skill. Hue: {self.hue}.")

        hue_tol = 25

        lower_HSV = np.array([max(0, self.hue - hue_tol), 50, 50])
        upper_HSV = np.array([min(179, self.hue + hue_tol), 255, 255])
        distances = self.lidars.read()
        to_the_left = distances[0] < distances[7]

        start_time = time.time()
        while time.time() - start_time < self.timeout:

            # 1. Camera Snapshot & HSV Detection
            frame = self.camera.read()
            distances = self.lidars.read()
            if frame is None:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_HSV, upper_HSV)

            # 2. Locate Green Centroid
            moments = cv2.moments(mask)
            frame_center_x = frame.shape[1] // 2

            if moments["m00"] > 100:  # Green detected (filtering noise)
                self.wheels.set_parameters([0.0, 0.0])
                print(f"[AGENT] {self.__class__.__name__} finished successfully.")
                return True
            else:
                # No green found: continue searching by rotating
                if to_the_left:
                    self.wheels.set_parameters([ -self.velocity, self.velocity])
                else:
                    self.wheels.set_parameters([ self.velocity, -self.velocity])

            time.sleep(0.01)

        # Timeout reached
        self.wheels.set_parameters([0.0, 0.0])
        print(f"[AGENT] {class_name} failed. Timed out.")
        return False

class ClearThePath(Skill):
    '''
    Clears the path in front by turning and running away.
    '''
    def __init__(
        self,
        bus: Communicator,
        camera, 
        lidars, 
        wheels, 
        velocity: float, 
        min_dist: float,
        run_time: float,
        timeout: float = 15.0
    ):
        super().__init__()
        self.bus = bus
        self.camera = camera 
        self.lidars = lidars
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.run_time = run_time
        self.timeout = timeout

    def execute(self) -> bool:
        print(f"[AGENT] Executing {self.__class__.__name__} skill.")
        start_time = time.time()
        tolerance = 0.005
        # helpers
        def turn() -> bool:
            frame = self.camera.read()
            distances = self.lidars.read()
            distance_left = distances[0]
            distance_right = distances[7]
            if (( distances[0] > self.min_dist + tolerance ) and ( distances[7] > self.min_dist + tolerance )):
                return True
            if distance_left < distance_right:
                self.wheels.set_parameters([ - self.velocity, self.velocity])
            else:
                self.wheels.set_parameters([self.velocity, - self.velocity])
            while time.time() - start_time < self.timeout:
                frame = self.camera.read()
                distances = self.lidars.read()
                # turn until minimal value is on indexes 3 and 4
                min_idx = np.argmin(distances)
                clear_path = (( distances[0] > self.min_dist + tolerance ) and ( distances[7] > self.min_dist + tolerance ))
                if min_idx in [3, 4] and clear_path:
                    return True
            self.wheels.set_parameters([0, 0])
            print(f"[AGENT] {self.__class__.__name__} failed. Timed out. TURN")
            return False

        def move() -> bool:
            start_run = time.time()
            self.wheels.set_parameters([self.velocity, self.velocity])
            while time.time() - start_run < self.run_time:
                # sensors
                frame = self.camera.read()
                distances = self.lidars.read()
                if time.time() - start_time > self.timeout:
                    self.wheels.set_parameters([0, 0])
                    print(f"[AGENT] {self.__class__.__name__} failed. Timed out. MOVE")
                    return False
                clear_path = (( distances[0] > self.min_dist + tolerance ) and ( distances[7] > self.min_dist + tolerance ))
                if not clear_path:
                    return True if turn() else False
            return True


        # loops:::
        # sensors
        frame = self.camera.read()
        distances = self.lidars.read()
        # calculations
        ## all obstacles = bad
        if max(distances) < self.min_dist + tolerance:
            self.wheels.set_parameters([0, 0])
            print(f"[AGENT] {self.__class__.__name__} failed. Nowhere to run.")
            return False
        ## find clear path:
        if turn() and move():
            self.wheels.set_parameters([0,0])
            print(f"[AGENT] {self.__class__.__name__} finished successfully.")
            return True
        ## failed somewhere
        self.wheels.set_parameters([0, 0])
        print(f"[AGENT] {self.__class__.__name__} failed.")
        return False


class GoToTheColor(Skill):
    '''
    Approaches a specific colored object by Hue.
    '''
    def __init__(
        self,
        bus: Communicator,
        hue: int,
        camera, 
        lidars, 
        wheels, 
        velocity: float, 
        min_dist: float, 
        target_distance: float = 0.06,
        distance_tolerance: float = 0.01,
        timeout: float = 15.0
    ):
        super().__init__()
        self.bus = bus
        self.hue = hue
        self.camera = camera 
        self.lidars = lidars
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.timeout = timeout
        self.center_tolerance = 10
        self.hue_tolerance = 25


    def _get_color_x_center(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # 2. Handle Hue wrapping (0-179 range in OpenCV)
        lower1, upper1 = max(0, self.hue - self.hue_tolerance), min(179, self.hue + self.hue_tolerance)
        mask = cv2.inRange(hsv, np.array([lower1, 50, 50]), np.array([upper1, 255, 255]))
    
        # Wrap around 0 (e.g., target hue is 5, we need 175-179 as well)
        if self.hue - self.hue_tolerance < 0:
            lower2 = 180 + (self.hue - self.hue_tolerance)
            mask2 = cv2.inRange(hsv, np.array([lower2, 50, 50]), np.array([179, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
        
        # Wrap around 179 (e.g., target hue is 175, we need 0-5 as well)
        elif self.hue + self.hue_tolerance > 179:
            upper2 = (self.hue + self.hue_tolerance) - 180
            mask2 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([upper2, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
    
        # Find the center of the colored object
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return None
        return int(M["m10"] / M["m00"])
    
    def conditions_check(self) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__} needs checks implementations. No conditions_check")

    def effects_check(self) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__} needs checks implementations. No effects_check")

    def execute(self) -> bool:
        print(f"[AGENT] Executing {self.__class__.__name__} skill. Hue: {self.hue}.")
        start_time = time.time()
        # read sensors
        frame = self.camera.read()
        distances = self.lidars.read()
        if frame is None: 
            return False
        # ask supervisor
        if not self.conditions_check():
            self.wheels.set_parameters([0, 0])
            print(f"[AGENT] {self.__class__.__name__} failed. Starting conditions not met.")
            return False
        # make calculations
        image_center = frame.shape[1] / 2
        color_center = self._get_color_x_center(frame)
        if color_center is None:
            self.wheels.set_parameters([0, 0])
            print(f"[AGENT] {self.__class__.__name__} failed. No target color in sight.")
            return False
        # loop
        while time.time() - start_time < self.timeout:
            # read sensors
            frame = self.camera.read()
            distances = self.lidars.read()
            if frame is None: 
                continue
            # make calculations
            image_center = frame.shape[1] / 2
            color_center = self._get_color_x_center(frame)
            if color_center is None:
                self.wheels.set_parameters([0, 0])
                print(f"[AGENT] {self.__class__.__name__} failed. No target color in sight.")
                return False
            front_dist = min(distances[0], distances[7])

            # check if we succeed
            at_target_distance = ( abs(front_dist - self.target_distance) < self.distance_tolerance )
            centered = ( abs(color_center - image_center) < self.center_tolerance )
            
            success = at_target_distance and centered
            
            if success:
                self.wheels.set_parameters([0, 0])
                # ask supervisor
                if not self.effects_check():
                    print(f"[AGENT] {self.__class__.__name__} failed. Effects conditions not met.")
                    return False
                print(f"[AGENT] {self.__class__.__name__} finished successfully.")
                return True

            # check if we can move freely
            clear_path = ( distances[0] >= self.min_dist and distances[7] >= self.min_dist )
            if not clear_path:
                self.wheels.set_parameters([0, 0])
                print(f"[AGENT] {self.__class__.__name__} failed. Obstacle ahead.")
                return False

            # 2 bools so 4 cases but 2 actions
            def centering():
                if color_center < image_center:
                    self.wheels.set_parameters([self.velocity * 0.1, self.velocity])
                elif color_center > image_center:
                    self.wheels.set_parameters([self.velocity, self.velocity * 0.1])
                time.sleep(0.01)
                
            def aproaching():
                if front_dist < self.target_distance:
                    self.wheels.set_parameters([ - self.velocity, - self.velocity])
                elif front_dist > self.target_distance:
                    self.wheels.set_parameters([self.velocity, self.velocity])
                time.sleep(0.01)

            # 1. == success
            # 2.
            if not at_target_distance and centered: aproaching()
            # 3.
            elif at_target_distance and not centered: centering()
            # 4.
            elif not at_target_distance and not centered: centering()


        self.wheels.set_parameters([0, 0])
        print(f"[AGENT] {self.__class__.__name__} failed. Timed out.")
        return False

class GoToTheDoor(GoToTheColor):
    def conditions_check(self) -> bool:
        return True

    def effects_check(self) -> bool:
        door_zone = self.bus.call_service(f"/supervisor/ask/door_zone")
        return door_zone

class GoToTheGoal(GoToTheColor):
    def conditions_check(self) -> bool:
        if self.bus.call_service(f"/supervisor/ask/room_number") == 2:
            return True
        return False

    def effects_check(self) -> bool:
        return self.bus.call_service(f"/supervisor/ask/goal_zone")

class GoThroughTheDoor(Skill):
    '''
    Opens door, any color.
    '''
    def __init__(
        self,
        bus: Communicator,
        camera: CameraSensor, 
        lidars: VirtualSensorArray, 
        wheels: VirtualActuator, 
        velocity: float, 
        min_dist: float,
        timeout: float = 15.0
    ):
        super().__init__()
        self.bus = bus
        self.camera = camera 
        self.lidars = lidars
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.timeout = timeout
        self.start_time = None

    def execute(self)-> bool:
        print(f"[AGENT] Executing {self.__class__.__name__} skill")
        # Ask supervisor if we can start:
        if not self.bus.call_service(f"/supervisor/ask/door_zone"):
            self.wheels.set_parameters([0, 0])
            print(f"[AGENT] {self.__class__.__name__} failed. Not in the door zone.")
            return False
        init_room = self.bus.call_service(f"/supervisor/ask/room_number")

        # Start moving forward
        self.wheels.set_parameters([self.velocity, self.velocity])
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            # sensors
            distances = self.lidars.read()
            frame = self.camera.read()
            room = self.bus.call_service(f"/supervisor/ask/room_number")
            zone = self.bus.call_service(f"/supervisor/ask/door_zone")
            # calculations
            front = min(distances[0], distances[7])
            clear_path = front > self.min_dist
            # logic
            if not room == init_room and not zone and clear_path:
                self.wheels.set_parameters([0.0, 0.0])
                print(f"[AGENT] {self.__class__.__name__} finished successfully.")
                return True 
            time.sleep(0.05)
            
        self.wheels.set_parameters([0.0, 0.0])
        print(f"[AGENT] {self.__class__.__name__} failed. Timed out.")
        return False 

