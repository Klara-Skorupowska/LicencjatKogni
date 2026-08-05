# a parent class for virtual actuators
from turtle import left

from communicator import Communicator
class VirtualActuator():
    def __init__(self, bus: Communicator):
        self.bus = bus

    def set_parameters(self):
        raise NotImplementedError("The set_parameters() method must be implemented in the subclass.")
        # here you must sent the value to the real actuator, so that the real actuator can get it from the bus

class VirtualActuatorArray(VirtualActuator):
    def __init__(self, bus: Communicator, actuators: list[VirtualActuator]):
        super().__init__(bus)
        self.actuators = actuators
    def set_parameters(self, value):
        for actuator in self.actuators:
            actuator.set_parameters(value)

class WheelsActuator(VirtualActuator):
    def __init__(self, bus: Communicator):
        super().__init__(bus)
        
    def set_parameters(self, values):
        '''
        values = [left, right]
        '''
        left_wheel, right_wheel = values
        wheel_msg = {"left": left_wheel, "right": right_wheel}
        self.bus.publish("/cmd/wheels", wheel_msg)

