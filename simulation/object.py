### parent class for all environment objects
import xacro
import numpy as np
import pybullet as p
from communicator import Communicator

class Object():
    def __init__(self):
        self.id = None
        self.anchored = True
        self.model_name = "cube"
        self.initial_position = [0, 0, 0]
        self.initial_orientation = [0, 0, 0, 1]
        self.sensors = None
        self.actuators = None
        self.bus = None

    def create_urdf_from_xacro(self, modelname):
        xacro_path = f"models/{modelname}.xacro"
        urdf_path = f"models/urdfs/{modelname}.urdf"
        doc = xacro.process_file(xacro_path)
        with open(urdf_path, "w") as f:
            f.write(doc.toxml())

        return urdf_path

    def setID(self, id):
        ''' all sensors and actuators need to be updated here'''
        self.id = id

    def set_communicator(self, bus: Communicator):
        # set communicator for sensors and actuators if they exist
        if self.sensors is not None:
            self.bus = bus
            for sensor in self.sensors.values():
                if isinstance(sensor, (list, tuple, np.ndarray)):
                    for s in sensor:
                        s.set_communicator(self.bus)
                else:
                    sensor.set_communicator(self.bus)
        if self.actuators is not None:
            self.bus = bus
            for actuator in self.actuators.values():
                if isinstance(actuator, list):
                    for a in actuator:
                        a.set_communicator(self.bus)
                else:
                    actuator.set_communicator(self.bus)

    def _joint_name_to_index(self):
        # Map all joint names to their indices
        joint_name_to_index = {}
        for i in range(p.getNumJoints(self.id)):
            joint_info = p.getJointInfo(self.id, i)
            joint_name = joint_info[1].decode('utf-8') 
            joint_name_to_index[joint_name] = i
        return joint_name_to_index

    def setup(self):
        '''
        for any additional pybullet set up needed
        '''
        pass

    def sense(self):
        raise NotImplementedError("The sense() method must be implemented in the subclass.")
    def act(self):
        raise NotImplementedError("The act() method must be implemented in the subclass.")
    def get_value(self):
        raise NotImplementedError("The get_value() method must be implemented in the subclass.")

