## main file ###

# --- imports ---
import threading
import time
import pybullet as p

from simulation import *
from agent import *
from supervisor import *
from communicator import Communicator
from visualizer import PipelineVisualizer


# --- let's go ---
def main():
    print("Starting...")
    print("Open communication")
    bus = Communicator()
    ### ---- CHOOSE PARAMETERS HERE ---- ###
    ## agent ##
    agent = TheAgent(bus)
    ## simulation ##
    robot = RealRobot(bus)
    arena = Arena()
    ### ---- ------ ---------- ---- ---- ###
    # event for waiting for full simulation initialization
    simulation_ready_event = threading.Event()
    # event for waiting for full suprevisor and data initialization
    supervisor_ready_event = threading.Event()
    # event for closing everything
    stop_event = threading.Event()
    # create threads
    physics_thread = threading.Thread(target=simulation_loop, args=(bus, simulation_ready_event, robot, arena, stop_event, ), daemon=True)
    agent_thread = threading.Thread(target=agent_loop, args=(bus, agent, stop_event, ), daemon=True)
    supervisor_thread = threading.Thread(target=supervisor_loop, args=(bus, supervisor_ready_event, stop_event,), daemon=True)
    # start 
    print("Start Nodes")
    physics_thread.start()
    simulation_ready_event.wait()
    supervisor_thread.start()
    supervisor_ready_event.wait()
    agent_thread.start()

    # end 
    agent_thread.join()
    physics_thread.join()
    supervisor_thread.join()

    # pictures
    runner = PipelineVisualizer(logs_root="logs")
    runner.run_all()

    print("All threads closed cleanly. Exiting program.")

if __name__ == "__main__":
    main()

