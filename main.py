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
    # event for closin' everything
    stop_event = threading.Event()
    # create threads
    physics_thread = threading.Thread(target=simulation_loop, args=(bus, simulation_ready_event, robot, arena, stop_event, ), daemon=True)
    agent_thread = threading.Thread(target=agent_loop, args=(bus, agent, stop_event, ), daemon=True)
    # start 
    print("Start Nodes")
    physics_thread.start()
    simulation_ready_event.wait()
    agent_thread.start()

    # end 
    agent_thread.join()
    physics_thread.join()
    print("All threads closed cleanly. Exiting program.")

if __name__ == "__main__":
    main()

