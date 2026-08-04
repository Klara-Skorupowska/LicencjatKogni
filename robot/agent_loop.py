from communicator import Communicator
from robot import SimpleAgent, SkilledAgent, TheAgent

def agent_loop(bus:Communicator):
    print("Loading robot...")
    robot = TheAgent(bus) ### choose your agent here
    print("Robot loaded.")
    robot.run() # the brain infinite loop