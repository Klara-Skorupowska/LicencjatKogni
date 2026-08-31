from .stats_agent import StatsAgent
from communicator import Communicator
from .core import Agent
from .stats_agent import *
import threading

def agent_loop(bus:Communicator, agent:Agent, stop_event: threading.Event):
    print("[AGENT] Loading brain logic...")
    #stats = [NetworkLogger(agent)]
    robot = agent
    #robot.add_stats(stats)
    print("[AGENT] Brain logic loaded.")
    robot.run() # the brain = (in)finite loop
    print("[AGENT] Robot gave up.")
    stop_event.set()