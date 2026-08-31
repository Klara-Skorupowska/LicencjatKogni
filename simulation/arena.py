### Simulated arena

from .object import Object
import pybullet as p

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

    def register_services(self):
        self.bus.register_service(f"/arena/give_id", self.give_id)


    def setup(self):
        door_hinge_index = self._joint_name_to_index().get("door_hinge")
        p.setJointMotorControl2(
            bodyUniqueId=self.id,
            jointIndex=door_hinge_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,  # closed
            force=0.01           # pushing force
        )

    