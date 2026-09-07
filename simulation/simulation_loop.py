from communicator import Communicator
import time
import pybullet as p
from .arena import Arena
from .environment import Environment
from . real_robot import RealRobot
from .object import Object
import threading

def simulation_loop(bus: Communicator, setup_complete_event, robot: RealRobot, arena: Arena, stop_event: threading.Event, objects:list[Object]=[]):
    """Runs continuously at 240Hz, completely independent of the agent."""
    print("[Simulation] Starting world clock")
    print("[Simulation] Loading environment")
    world = Environment([robot, arena] + objects)
    time_step = 1.0 / 240.0
    world.setup(bus, time_step)
    setup_complete_event.set()
    print("[Simulation] Environment loaded.")
    
    try:
        while not stop_event.is_set():
            p.stepSimulation()
            time.sleep(time_step)

    except KeyboardInterrupt:
        print("[Simulation] Simulation stopped by user.")
    finally:
        world.close()
        print("[Simulation] World closed safely.")
