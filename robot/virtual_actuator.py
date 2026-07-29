# a parent class for virtual actuators
from communicator import Communicator
class VirtualActuator():
    def __init__(self, bus: Communicator):
        self.bus = bus

    def set_parameters(self):
        raise NotImplementedError("The set_parameters() method must be implemented in the subclass.")
        # here you must sent the value to the real actuator, so that the real actuator can get it from the bus

class WheelsActuator(VirtualActuator):
    def __init__(self, bus: Communicator):
        super().__init__(bus)
        
    def set_parameters(self, left_wheel, right_wheel):
        wheel_msg = {"left": left_wheel, "right": right_wheel}
        self.bus.publish("/cmd/wheels", wheel_msg)

