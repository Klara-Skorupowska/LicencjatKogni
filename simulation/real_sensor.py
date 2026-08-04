# a parent class for 'real' (simulated) sensors 

from communicator import Communicator
import pybullet as p
import math
import numpy as np
import cv2

class RealSensor():
    def __init__(self):
        self.bus = None
        self.robot_id = None

    def __init__(self, bus: Communicator):
        self.bus = bus
        self.robot_id = None

    def set_communicator(self, bus: Communicator):
        self.bus = bus

    def sense():
        ''' getting data from the environment '''
        raise NotImplementedError("The sense() method must be implemented in the subclass.")  
    def get_data():
        ''' getting data from within the class '''
        raise NotImplementedError("The get_data() method must be implemented in the subclass.")  

class RealSensorArray(RealSensor):
    def __init__(self, bus: Communicator, sensors: list[RealSensor]):
        super().__init__(bus)
        self.sensors = sensors
    def sense(self):
        ''' getting data from the environment '''
        return [sensor.sense() for sensor in self.sensors]
    def get_data(self):
        ''' getting data from within the class '''
        return [sensor.get_data() for sensor in self.sensors]

class CameraSensor(RealSensor):
    def __init__(self, bus: Communicator, res: tuple[int, int, int]):
        '''
        bus - communicator,
        res (resolution) - [heigth, width, channels]
        '''
        super().__init__(bus)
        self.robot_id = None
        self.height, self.width, self.channels = res
        if self.channels not in (1, 3, 4): raise ValueError(f"Incorrect number of channels: {self.channels}. Expected 1, 3, or 4.")
        self.frame = None

        self.bus.register_service("/sensor/camera/sense", self.sense)

    def sense(self, request=None):
        ''' This is the Service Handler. '''
        pos, orient = p.getBasePositionAndOrientation(self.robot_id)
        matrix = p.getMatrixFromQuaternion(orient)
        forward = [matrix[0], matrix[3], matrix[6]]
        up = [matrix[2], matrix[5], matrix[8]]
        
        cam_pos = [pos[0] + forward[0]*0.02, pos[1] + forward[1]*0.02, pos[2] + 0.005]
        target = [cam_pos[0] + forward[0], cam_pos[1] + forward[1], cam_pos[2] + forward[2]]

        view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos, cameraTargetPosition=target, cameraUpVector=up)
        proj_matrix = p.computeProjectionMatrixFOV(90, self.width / self.height, 0.001, 10.0)
        
        img_arr = p.getCameraImage(self.width, self.height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        rgba = np.reshape(img_arr[2], (self.height, self.width, 4)).astype(np.uint8) # all 4 channels

        # Convert image format based on channels count
        if self.channels == 1: # Grayscale
            self.frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        elif self.channels == 3: # Standard RGB (OpenCV expects BGR format)
            self.frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        elif self.channels == 4: # Keep Alpha channel (BGRA format)
            self.frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        else:
            raise ValueError(f"Unsupported channel count: {self.channels}")

        return self.frame

    def get_data(self):
        return self.frame

class LidarSensor(RealSensor):
    def __init__(self, bus: Communicator, lidar_direction, sensor_range):
        '''
        lidar_direction: - in degrees, 0 is forward, 90 is left, 180 is backward, 270 is right
        '''
        super().__init__(bus)
        self.robot_id = None
        self.lidar_direction = lidar_direction
        self.sensor_range = sensor_range
        self.value = None

        self.bus.register_service(f"/sensor/lidar_{self.lidar_direction}/sense", self.sense)

    def sense(self, request=None):
        ''' This is the Service Handler. It runs ONLY when another node calls bus.call_service("/sensor/lidar_num/sense"). '''
        pos, orient = p.getBasePositionAndOrientation(self.robot_id)
        matrix = p.getMatrixFromQuaternion(orient)
        angle_rad = math.radians(self.lidar_direction)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        sensor_dir = [
            matrix[0] * cos_a + matrix[1] * sin_a,
            matrix[3] * cos_a + matrix[4] * sin_a,
            matrix[6] * cos_a + matrix[7] * sin_a
        ]
        start = [pos[0] + sensor_dir[0] * 0.036, pos[1] + sensor_dir[1] * 0.036, pos[2]]
        end = [start[0] + sensor_dir[0] * self.sensor_range, start[1] + sensor_dir[1] * self.sensor_range, start[2]]
        ray = p.rayTest(start, end)
        self.value = ray[0][2] * self.sensor_range
        return self.value

    def get_data(self):
        return self.value