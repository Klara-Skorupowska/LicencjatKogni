### Simulated arena

from .object import Object

class Arena(Object):
    def __init__(self):
        super().__init__()  
        self.model_name = "arena"
        self.initial_position = [0, 0, 0]
        self.initial_orientation = [0, 0, 0, 1] 

    def sense(self):
        print("EnvArena: sense() method called, but no sensors to sense.")
    def act(self):
        print("EnvArena: act() method called, but no actuators to act.")