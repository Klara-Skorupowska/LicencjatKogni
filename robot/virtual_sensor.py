# a parent class for virtual sensors

from turtle import width

from communicator import Communicator


class VirtualSensor():
    def __init__(self, bus: Communicator):
        self.bus = bus

    def read(self): # from the bus
        raise NotImplementedError("The read() method must be implemented in the subclass.")

class VirtualSensorArray(VirtualSensor):
    def __init__(self, bus: Communicator, sensors: list[VirtualSensor]):
        super().__init__(bus)
        self.sensors = sensors
    def read(self):
        return [sensor.read() for sensor in self.sensors]

class LidarSensor(VirtualSensor):
    def __init__(self, bus: Communicator, lidar_direction):
        '''
        lidar_direction: - in degrees, 0 is forward, 90 is left, 180 is backward, 270 is right
        '''
        super().__init__(bus)
        self.lidar_direction = lidar_direction
        self.value = None

    def read(self):
        self.value = self.bus.call_service(f"/sensor/lidar_{self.lidar_direction}/sense")
        return self.value

class CameraSensor(VirtualSensor):
    def __init__(self, bus: Communicator):
        '''
        res (resolution) - [heigth, width, channels]
        '''
        super().__init__(bus)
        self.frame = None

    def read(self):
        self.frame = self.bus.call_service("/sensor/camera/sense")
        return self.frame
