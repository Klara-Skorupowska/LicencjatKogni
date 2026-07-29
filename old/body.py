import math
import pybullet as p
import numpy as np
import cv2

class Body(object):
    ''' Parent class representing a body: URDF file and extra sensors etc.'''
    def __init__(self, name, model, observations_map, actuators_map):
        self.name = name
        self.model = model
        self.observations_map = observations_map
        self.actuators_map = actuators_map
        self.id = None # to be set by robot when loading URDF

    def get_observations(self):
        ''' To be implemented by child classes. It should return data for brain.'''
        pass

    def set_actuators(self, data):
        ''' To be implemented by child classes. It should take data from brain and set actuators accordingly.'''
        pass


class Epuck2Body(Body):
    ''' the body of E-puck 2 robot, with all their inputs/actuators (relevant to task)'''
    def __init__(self, name):
        model = "epuck2" # name of xacro file in models folder, will be converted to urdf by helper function in main loop
        
        # parameters
        self.sensor_angles = [17, 50, 90, 150, 210, 270, 310, 343]
        self.sensor_range = 0.1

        self.max_force = 10.0 
        self.max_speed = 20.0

        self.cam_width = 320
        self.cam_height = 240
        self.cam_channels = 3

        observations_map ={
            "dist_front": None, 
            "dist_left": None, 
            "dist_right": None, 
            "video": np.abs(np.zeros((self.cam_height, self.cam_width, self.cam_channels), dtype=np.uint8))
        }
        actuators_map = {
            "left_motor": None, 
            "right_motor": None
        }
        super().__init__(name, model, observations_map, actuators_map)


    def get_sensor_readings(self):
        pos, orient = p.getBasePositionAndOrientation(self.id)
        matrix = p.getMatrixFromQuaternion(orient)
        readings = []
        for angle_deg in self.sensor_angles:
            angle_rad = math.radians(angle_deg)
            sensor_dir = [
                matrix[0] * math.cos(-angle_rad) + matrix[1] * math.sin(-angle_rad),
                matrix[3] * math.cos(-angle_rad) + matrix[4] * math.sin(-angle_rad),
                matrix[6] * math.cos(-angle_rad) + matrix[7] * math.sin(-angle_rad)
            ]
            start = [pos[0] + sensor_dir[0]*0.036, pos[1] + sensor_dir[1]*0.036, pos[2]]
            end = [start[0] + sensor_dir[0]*self.sensor_range, start[1] + sensor_dir[1]*self.sensor_range, start[2]]
            ray = p.rayTest(start, end)
            readings.append(ray[0][2])
        return readings

    def get_camera_image(self):
        pos, orient = p.getBasePositionAndOrientation(self.id)
        matrix = p.getMatrixFromQuaternion(orient)
        forward = [matrix[0], matrix[3], matrix[6]]
        up = [matrix[2], matrix[5], matrix[8]]
        
        cam_pos = [pos[0] + forward[0]*0.02, pos[1] + forward[1]*0.02, pos[2] + 0.005]
        target = [cam_pos[0] + forward[0], cam_pos[1] + forward[1], cam_pos[2] + forward[2]]

        view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos, cameraTargetPosition=target, cameraUpVector=up)
        proj_matrix = p.computeProjectionMatrixFOV(90, self.cam_width/self.cam_height, 0.001, 10.0)
        
        img_arr = p.getCameraImage(self.cam_width, self.cam_height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        rgba = np.reshape(img_arr[2], (self.cam_height, self.cam_width, 4)).astype(np.uint8)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def set_motors(self, v_left, v_right):
        v_left = int(v_left * self.max_speed)
        v_right = int(v_right * self.max_speed)
        p.setJointMotorControl2(self.id, 0, p.VELOCITY_CONTROL, targetVelocity=v_left, force=self.max_force)
        p.setJointMotorControl2(self.id, 1, p.VELOCITY_CONTROL, targetVelocity=v_right, force=self.max_force)

    def get_observations(self):
        
        readings = self.get_sensor_readings()

        self.observations_map["dist_front"] = np.min([readings[0], readings[7]])
        self.observations_map["dist_left"] = np.min([readings[5], readings[6]])
        self.observations_map["dist_right"] = np.min([readings[1], readings[2]])
        self.observations_map["video"] =  self.get_camera_image()

        return self.observations_map

    def set_actuators(self, actuators_map):
        self.actuators_map = actuators_map
        v_left = actuators_map["left_motor"]
        v_right = actuators_map["right_motor"]
        self.set_motors(v_left, v_right)