from communicator import Communicator
import time
import pybullet as p
from simulation import Environment, RealRobot, Arena, StatusBoard, DataLogger

def simulation_loop(bus: Communicator, setup_complete_event):
    """Runs continuously at 240Hz, completely independent of the agent."""
    print("[Physics] Starting world clock...")
    print("Loading environment...")
    robot = RealRobot(bus) ### choose your robot here)
    world = Environment([robot, Arena()], [StatusBoard("Robot_1", robot), DataLogger(robot, log_dir="./logs")]) ### choose your world here (arena, stats)
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
