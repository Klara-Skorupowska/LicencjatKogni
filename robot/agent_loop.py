from communicator import Communicator
from robot import SimpleAgent, SkilledAgent

def agent_loop(bus:Communicator):
    print("Loading robot...")
    robot = SkilledAgent(bus) ### choose your agent here
    print("Robot loaded.")
    robot.run() # the brain infinitive loop