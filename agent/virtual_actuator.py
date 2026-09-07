# a parent class for virtual actuators
import time

from communicator import Communicator
class VirtualActuator():
    def __init__(self, bus: Communicator):
        self.bus = bus
        self.value = None
        self.register_services()

    def set_parameters(self):
        raise NotImplementedError("The set_parameters() method must be implemented in the subclass.")
        # here you must sent the value to the real actuator, so that the real actuator can get it from the bus

    def register_services(self):
        raise NotImplementedError("The register_services() method must be implemented in the subclass.")
        # here you must register services, such as send_value
    
    def send_value(self, request=None):
        return self.value

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

    def register_services(self):
        self.bus.register_service(f"/wheels/ask/speed", self.send_value)
        
    def set_parameters(self, values):
        '''
        values = [left, right]
        '''
        self.value = values
        left_wheel, right_wheel = values
        wheel_msg = {"left": left_wheel, "right": right_wheel}
        self.bus.publish("/cmd/wheels", wheel_msg)
        time.sleep(0.01)

