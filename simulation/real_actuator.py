# a parent class for 'real' actuator, which is a wrapper for the real actuator in the environment

import pybullet as p
from communicator import Communicator

class RealActuator():
    def __init__(self):
        self.bus = None
        self.robot_id = None

    def __init__(self, bus: Communicator):
        self.bus = bus
        self.robot_id = None

    def set_communicator(self, bus: Communicator):
        self.bus = bus

    def set_robot_id(self, robot_id):
        self.robot_id = robot_id

    def act(self):
        ''' used for changing parameters manually '''
        raise NotImplementedError("The act() method must be implemented in the subclass.") 

class RealActuatorArray(RealActuator):
    def __init__(self, bus: Communicator, actuators: list[RealActuator]):
        super().__init__(bus)
        self.actuators = actuators

    def set_robot_id(self, robot_id):
        for act in self.actuators:
            act.set_robot_id(robot_id)

    def act(self, value):
        ''' used for changing parameters manually. each one get the same value(s)'''
        for actuator in self.actuators:
            actuator.act(value)

# wheels
class WheelsActuator(RealActuator):
    def __init__(self, bus: Communicator):
        super().__init__(bus)
        self.left_wheel_joint_index = None
        self.right_wheel_joint_index = None
        self.max_velocity = 20.0
        self.max_force = 2.0 
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0

        self.bus.subscribe("/cmd/wheels", self.wheels_callback)

    def set_robot_id(self, id):
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

    def act(self, values):
        ''' manual way of changing velocities.
        values - a list of 2 floats: [left_wheel_velocity, right_wheel_velocity]
        '''
        left_wheel_velocity, right_wheel_velocity = values
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

    def get_value(self):
        ''' returns the current velocities of the wheels '''
        return [self.left_wheel_velocity, self.right_wheel_velocity]