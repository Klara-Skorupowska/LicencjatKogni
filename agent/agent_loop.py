from .stats_agent import StatsAgent
from communicator import Communicator
from .core import Agent
from .stats_agent import *

def agent_loop(bus:Communicator, agent:Agent):
    print("Loading brain logic...")
    stats = [NetworkLogger(agent)]
    robot = agent
    robot.add_stats(stats)
    print("Brain logic loaded.")
    robot.run() # the brain = infinite loop