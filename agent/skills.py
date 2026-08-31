# all basic skills
import time
import math

from .virtual_actuator import *
from .virtual_sensor import *

class Skill():
    '''
    parent class for skills, each skill have set of sensors/actuators it requires
    '''
    def __init__(self):
        pass

    def execute(self):
        ''' executes the skill from start to finish and returns if the execution was completed '''
        raise NotImplementedError("The execute() method must be implemented in the subclass.") 



### --- EXTRA CALCULATIONS FOR EPUCK2   --- ###
T_CONST = math.pi*0.052/(0.02)  # based on epuck2 (wheel radius and distance between them), needed to be devided by the velocity to get the time needed to turn 360 deg
### ---                                 --- ###
class MoveForward(Skill):
    ''' moves forward until it aproaches a wall '''
    def __init__(self, lidar_forward_left: LidarSensor, lidar_forward_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float, time: float = 5.0):
        '''
        lidar_forward_left - a lidar with angle (0-30) deg
        lidar_forward_right - a lidar with angle (330-360) deg
        wheels - a set of wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity 
        time - the time of moving forward (in seconds)
        '''
        super().__init__()
        self.lidar_forward_left = lidar_forward_left
        self.lidar_forward_right = lidar_forward_right
        self.wheels = wheels

        self.velocity = velocity
        self.min_dist = min_dist
        self.time = time

    def execute(self):
        print("[AGENT] Executing MoveForward skill")
        self.wheels.set_parameters([self.velocity, self.velocity])
        start_time = time.time()
        
        while time.time() - start_time < self.time:
            # 1. Take a snapshot of the sensors
            front_left = self.lidar_forward_left.read()
            front_right = self.lidar_forward_right.read()

            # 2. Check if we have approached a wall
            if front_left < self.min_dist or front_right < self.min_dist:
                self.wheels.set_parameters([0.0, 0.0]) # halt
                return True # completed moving forward until hitting a wall
                
            # 3. Yield to the CPU
            time.sleep(0.01)
            
        # If we reach this point, the timeout expired before finding a wall = no need for halting, as we are still moving forward
        return True  # completed moving forward without hitting a wall

class Turn(Skill):
    '''
    right turn for set time
    '''
    def __init__(self, direction: str, wheels: WheelsActuator, velocity: float, time: float = 1.0):
        '''
        direction - direction of turning: 'right' or 'left'
        lidar_forward_left - a lidar with angle (0-30) deg
        lidar_forward_right - a lidar with angle (330-360) deg
        wheels - a set of 2 wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity 
        '''
        super().__init__()
        self.direction = direction
        self.wheels = wheels

        self.velocity = velocity
        self.time = time

    def execute(self):
        print(f"[AGENT] Executing Turn skill. Direction {self.direction}")
        left = 1
        right = 1
        if self.direction == 'right':
            right = -1
        elif self.direction == 'left':
            left = -1
        else:
            return False # todo raise

        self.wheels.set_parameters([left*self.velocity, right*self.velocity])
        start_time = time.time()
        while time.time() - start_time < self.time:
            time.sleep(0.01)  # Prevent CPU spinning

        self.wheels.set_parameters([0.0, 0.0]) # halt
        return True

class TurnAway(Skill):
    '''
    turn away from the obstacle ahead. Turning continues until the obstacle is at c. 120 deg (left or right) from the robot. 
    Skill fails if after the whole circle there is still something in front of the robot
    '''
    def __init__(self, lidar_front_left: LidarSensor, lidar_front_right: LidarSensor, lidar_back_left: LidarSensor, lidar_back_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float):
        '''
        lidar_front_left - a lidar with angle (0-30) deg
        lidar_front_right - a lidar with angle (330-360) deg
        lidar_back_left - a lidar with angle (90-120) deg
        lidar_back_right - a lidar with angle (240-270) deg
        wheels - a set of 2 wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity - the speed at which the robot should move
        '''
        super().__init__()
        self.lidar_front_left = lidar_front_left
        self.lidar_front_right = lidar_front_right
        self.lidar_back_left = lidar_back_left
        self.lidar_back_right = lidar_back_right
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist

    def execute(self):
        print("[AGENT] Executing TurnAway skill.")
        # 1. Find nearest obstacle
        front_left= self.lidar_front_left.read()
        front_right = self.lidar_front_right.read()

        if front_left < front_right:
            turn_direction = 'right'
            target_distance = front_left
            # If turning right (clockwise), the obstacle slides down the left side (CCW relative to robot)
            front_tracker = self.lidar_front_left
            back_tracker = self.lidar_back_left
        else:
            turn_direction = 'left'
            target_distance = front_right
            # If turning left (CCW), the obstacle slides down the right side (CW relative to robot)
            front_tracker = self.lidar_front_right
            back_tracker = self.lidar_back_right

        # 2. Initiate turn
        if turn_direction == 'right':
            self.wheels.set_parameters([self.velocity, -self.velocity])
        else:
            self.wheels.set_parameters([-self.velocity, self.velocity])

        full_circle_time = T_CONST / self.velocity
        epsilon = 0.01
        start_time = time.time()
        
        front_cleared = False 

        while time.time() - start_time < full_circle_time:
            # Snapshot required lidars
            current_front = front_tracker.read()
            current_back = back_tracker.read()
            
            current_front_left = self.lidar_front_left.read()
            current_front_right = self.lidar_front_right.read()

            # Step A: Check if we have rotated enough for the obstacle to leave the front sensor
            if not front_cleared:
                if abs(current_front - target_distance) >= epsilon:
                    front_cleared = True

            # Step B: If front is clear, check if the obstacle has reached the back sensor
            obstacle_is_behind = False
            if front_cleared:
                obstacle_is_behind = abs(current_back - target_distance) <= epsilon
                
            # Step C: Ensure our new path is safe
            path_is_clear = (current_front_left > self.min_dist) and (current_front_right > self.min_dist)

            # Step D: Exit condition
            if obstacle_is_behind and path_is_clear:
                self.wheels.set_parameters([0.0, 0.0]) # halt
                return True
                
            time.sleep(0.01)

        # Timeout reached (completed 360 turn without finding a clear path)
        self.wheels.set_parameters([0.0, 0.0])
        return False

class OpenDoor(Skill):
    '''
    Opens a door by moving forward until the door is detected by the left lidars 
    and not the front ones.
    '''
    def __init__(self, 
                 lidar_front_left: LidarSensor, 
                 lidar_front_right: LidarSensor, 
                 lidar_left: LidarSensor, 
                 wheels: WheelsActuator, 
                 velocity: float, 
                 min_dist: float, 
                 timeout: float = 10.0):
        super().__init__()
        self.lidar_front_left = lidar_front_left
        self.lidar_front_right = lidar_front_right
        self.lidar_left = lidar_left
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.timeout = timeout

    def execute(self):
        print("[AGENT] Executing OpenDoor skill")
        # Start moving forward
        self.wheels.set_parameters([self.velocity, self.velocity])
        
        front_left_dist = self.lidar_front_left.read()
        front_right_dist = self.lidar_front_right.read()

        if front_left_dist > self.min_dist and front_right_dist > self.min_dist:
            return False  # No door detected in front, cannot open

        start_time = time.time()
        while time.time() - start_time < self.timeout:
            # 1. Take a simultaneous snapshot of all required sensors
            left_dist = self.lidar_left.read()
            front_left_dist = self.lidar_front_left.read()
            front_right_dist = self.lidar_front_right.read()
            
            # 2. Evaluate the snapshot
            door_on_left = left_dist < self.min_dist
            front_is_clear = (front_left_dist > self.min_dist) and (front_right_dist > self.min_dist)
            
            if door_on_left and front_is_clear:
                self.wheels.set_parameters([0.0, 0.0]) # halt
                return True # there was a door and we opened it successfully
                
            # 3. Yield to the CPU
            time.sleep(0.01)
            
        # Timeout reached
        self.wheels.set_parameters([0.0, 0.0])
        return False # there was a door but we failed to open it in time


class Approach(Skill):
    ''' moves forward until it runs into an obstacle (or out of time) and position itself directly in front of it '''
    def __init__(self, lidar_front_left: LidarSensor, lidar_front_right: LidarSensor, wheels: WheelsActuator, velocity: float, min_dist: float, timeout: float, epsilon: float = 0.001):
        '''
        lidar_front_left - a lidar with angle (0-30) deg
        lidar_front_right - a lidar with angle (330-360) deg
        wheels - a set of 2 wheels
        min_dist - a minimal ditance from the obstacle (in meters)
        velocity - the speed at which the robot should move
        timeout - time after which robot stops and the aproach fails
        epsilon - acceptable distance error
        '''
        super().__init__()
        self.lidar_front_left = lidar_front_left
        self.lidar_front_right = lidar_front_right
        self.wheels = wheels
        self.velocity = velocity
        self.min_dist = min_dist
        self.timeout = timeout
        self.epsilon = epsilon

    def execute(self):
        print("[AGENT] Executing Approach skill")
        # Start moving forward
        self.wheels.set_parameters([self.velocity, self.velocity])
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            # 1. Take a simultaneous snapshot of all required sensors
            front_left_dist = self.lidar_front_left.read()
            front_right_dist = self.lidar_front_right.read()
            
            # 2. Evaluate the snapshot
            if front_left_dist <= self.min_dist or front_left_dist <= self.min_dist:  # if there is an obstacle ahead
                # - start to position itself in front of it = turn -
                if front_left_dist >= front_right_dist:
                    self.wheels.set_parameters([self.velocity/2, -self.velocity/2])
                else:
                    self.wheels.set_parameters([-self.velocity/2, self.velocity/2])
                start_time = time.time() # timeout for positioning
                while time.time() - start_time < self.timeout:
                    # snapshot of sensors
                    front_left_dist = self.lidar_front_left.read()
                    front_right_dist = self.lidar_front_right.read()
                    # check if we are facing the wall
                    if abs(front_right_dist - front_left_dist) < self.epsilon: 
                        self.wheels.set_parameters([0.0, 0.0])
                        return True # success: in front of the wall
                    time.sleep(0.01)
            # 3. Yield to the CPU
            time.sleep(0.01)
            
        # Timeout reached
        self.wheels.set_parameters([0.0, 0.0])
        return False # there was no obstacle that we could reach in time

class GoToTheDoor_OLD(Skill):
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
        time_step = 2.0
        decay_rate = 0.5
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
            if not (dir_new == dir_old): time_step = time_step * decay_rate
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
        print(f"[AGENT] Executing {self.__class__.__name__} skill")
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
                print(f"[AGENT] {self.__class__.__name__} failed. No door in sight.")
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
                        print(f"[AGENT] {self.__class__.__name__} finished successfully.")
                        return True
                    # or not
                    print(f"[AGENT] {self.__class__.__name__} failed. Obstacle ahead.")
                    self.wheels.set_parameters([0, 0])
                    return False
            else:
            # if not: turn to the wall/door so both front lidars have the same value 
                direction = 'RIGHT' if distances[0] > distances[7] else 'LEFT'
                if not self._turn_to_the_wall(direction): 
                    print(f"[AGENT]  {self.__class__.__name__} failed. Error while turning to the wall")
                    return False
                aligned = self._align()
                # if align == True, align the distance to target_distance -> return True
                # if false -> return False
                if aligned:
                    print(f"[AGENT] {self.__class__.__name__} finished successfully.")
                    return True
                else:
                    print(f"[AGENT] {self.__class__.__name__} failed. Aligment problem.")
                    return False
            # if timeout -> return False
        # Timeout reached
        self.wheels.set_parameters([0, 0])
        print(f"[AGENT] {self.__class__.__name__} failed. Timed out.")
        return False

class GoToTheGoal_OLD(Skill):
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
        print(f"[AGENT] Executing {self.__class__.__name__} skill")
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
                print(f"[AGENT] {self.__class__.__name__} failed. No goal in sight.")
                return False

            min_front = float(np.min([distances[0:2], distances[5:7]]))
            centered = abs(cx - frame.shape[1] // 2) < self.center_tolerance
            if abs(min_front - self.target_distance) < self.distance_tolerance:
                if centered:
                    self.wheels.set_parameters([0, 0])
                    print(f"[AGENT] {self.__class__.__name__} finished successfully.")
                    return True
            clear_path = ( float(np.min([distances[0:1], distances[6:7]])) > self.min_dist )
            if not clear_path and not centered:
                self.wheels.set_parameters([0, 0])
                print(f"[AGENT] {self.__class__.__name__} failed. Obstacle ahead.")
                return False

            err = (cx - frame.shape[1] // 2) / frame.shape[1]
            steer = err * self.velocity * 2
            v_left = self.velocity + steer
            v_right = self.velocity - steer
            self.wheels.set_parameters([v_left, v_right])

        # Timeout reached
        self.wheels.set_parameters([0, 0])
        print(f"[AGENT] {self.__class__.__name__} failed. Timed out.")
        return False
