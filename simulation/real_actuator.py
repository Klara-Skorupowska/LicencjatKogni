# a parent class for 'real' actuator, which is a wrapper for the real actuator in the environment

import pybullet as p
from communicator import Communicator

class RealActuator():
    def __init__(self):
        self.bus = None

    def act(self):
        ''' used for changing parameters manually '''
        raise NotImplementedError("The act() method must be implemented in the subclass.") 


# wheels
class WheelsActuator(RealActuator):
    def __init__(self, bus: Communicator):
        super().__init__()
        self.bus = bus
        self.robot_id = None
        self.left_wheel_joint_index = None
        self.right_wheel_joint_index = None
        self.max_velocity = 20.0
        self.max_force = 10.0 
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0

        self.bus.subscribe("/cmd/wheels", self.wheels_callback)

    def set_id(self, id):
        self.robot_id = id
        # joint indexes for actuators
        joint_name_to_index = {}
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')  # Byte string to string
            joint_name_to_index[joint_name] = i
        self.left_wheel_joint_index = joint_name_to_index.get("left_wheel_joint")
        self.right_wheel_joint_index = joint_name_to_index.get("right_wheel_joint")
    
    def wheels_callback(self, message):
        ''' a callback for chaning the wheels velocities in simulation '''
        left_velocity = message["left"]
        right_velocity = message["right"]
        self.left_wheel_velocity = max(-self.max_velocity, min(self.max_velocity, left_velocity))
        self.right_wheel_velocity = max(-self.max_velocity, min(self.max_velocity, right_velocity))
        p.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.left_wheel_joint_index, 
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=self.left_wheel_velocity,
            force = self.max_force
        )
        p.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.right_wheel_joint_index, 
            controlMode=p.VELOCITY_CONTROL, 
            targetVelocity=self.right_wheel_velocity,
            force = self.max_force
        )

    def act(self, left_wheel_velocity, right_wheel_velocity):
        ''' manual way of changing velocities '''
        p.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.left_wheel_joint_index, 
            controlMode=p.VELOCITY_CONTROL, 
            targetVelocity=left_wheel_velocity,
            force = self.max_force
        )
        p.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=self.right_wheel_joint_index, 
            controlMode=p.VELOCITY_CONTROL, 
            targetVelocity=right_wheel_velocity,
            force = self.max_force
        )