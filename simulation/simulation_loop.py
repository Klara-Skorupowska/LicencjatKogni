from communicator import Communicator
import time
import pybullet as p
from simulation import Environment, RealRobot, Arena, StatusBoard

def simulation_loop(bus: Communicator, setup_complete_event):
    """Runs continuously at 240Hz, completely independent of the agent."""
    print("[Physics] Starting world clock...")
    print("Loading environment...")
    world = Environment([RealRobot(bus), Arena()], [StatusBoard("Waldek", None, None)]) ### choose your world here
    time_step = 1.0 / 240.0
    world.setup(bus, time_step)
    setup_complete_event.set()
    print("Environment loaded.")
    
    while True:
        p.stepSimulation()
        # publish data from sensors (camera)
        time.sleep(time_step)
