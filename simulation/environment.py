### Environment class, ergo simulation, in RL unnecessary 
# it contains all pybullet stuff, like loading the arena, robot, etc.

import pybullet as p
import pybullet_data
import time
import cv2

from .object import Object
from .stats import *
from communicator import Communicator

class Environment():
    def __init__(self):
        self.objects = None
        self.stats = None

    def __init__(self, objects: list[Object] = [], stats: list[Stats] = []):
        self.objects = objects
        self.stats = stats

    def add_object(self, obj: Object):
        self.objects.append(obj)

    def add_stats(self, stat: Stats):
        self.stats.append(stat)

    def setup(self, com: Communicator, time_step: float):
        # Config physics and environment
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        p.resetDebugVisualizerCamera(1.0, 90, -120, [0,0,0])
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)

        # load all objects in the environment
        for obj in self.objects:
            urdf_path = obj.create_urdf_from_xacro(obj.model_name)
            obj_id = p.loadURDF(urdf_path, obj.initial_position, obj.initial_orientation, useFixedBase=obj.anchored)
            obj.setID(obj_id)
            obj.set_communicator(com)

    def update(self):
        for stat in self.stats:
            stat.update()

    def close(self):
        p.disconnect()
        for stat in self.stats:
            stat.close()

