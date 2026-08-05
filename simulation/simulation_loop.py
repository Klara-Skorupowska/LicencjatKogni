from communicator import Communicator
import time
import pybullet as p
from .arena import Arena
from .environment import Environment
from . real_robot import RealRobot
from .object import Object 
from .stats_sim import *

def simulation_loop(bus: Communicator, setup_complete_event, robot: RealRobot, arena: Arena, objects:list[Object]=[]):
    """Runs continuously at 240Hz, completely independent of the agent."""
    print("[Physics] Starting world clock...")
    print("Loading environment...")
    stats_sim = [StatusBoard("Robot_1", robot), DataLogger(robot, log_dir="./logs")]
    world = Environment([robot, arena] + objects, stats_sim)
    time_step = 1.0 / 240.0
    world.setup(bus, time_step)
    setup_complete_event.set()
    print("Environment loaded.")
    
    try:
        while True:
            p.stepSimulation()
            # update all stats every 1 s:
            if int(time.time()) % 1 == 0:
                world.update()
            time.sleep(time_step)

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        world.close()
