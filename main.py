## main file ###

# --- imports ---
import threading
import time
import pybullet as p

from agent.stats_agent import NetworkLogger
from simulation import *
from agent import *
from communicator import Communicator


# --- let's go ---
def main():
    print("Starting...")
    print("Open communication")
    bus = Communicator()
    ### ---- CHOOSE PARAMETERS HERE ---- ###
    # BRAIN # agent #
    agent = TheAgent(bus)
    # BODY # simulation #
    robot = RealRobot(bus)
    arena = Arena()
    ### ---- ------ ---------- ---- ---- ###
    # event for waiting for full simulation initialization
    simulation_ready_event = threading.Event()
    # create threads
    physics_thread = threading.Thread(target=simulation_loop, args=(bus, simulation_ready_event, robot, arena, ), daemon=True)
    agent_thread = threading.Thread(target=agent_loop, args=(bus, agent, ), daemon=True)
    # start 
    print("Start Nodes")
    physics_thread.start()
    simulation_ready_event.wait()
    agent_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down simulation...")
        p.disconnect()

if __name__ == "__main__":
    main()

