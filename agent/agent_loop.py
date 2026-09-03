from communicator import Communicator
from .core import Agent
import threading

def agent_loop(bus:Communicator, agent:Agent, stop_event: threading.Event):
    print("[AGENT] Loading brain logic...")
    robot = agent
    print("[AGENT] Brain logic loaded.")
    robot.run() # the brain = (in)finite loop
    print(f"[AGENT] Robot gave up in room {bus.call_service('/supervisor/ask/room_number')}")
    stop_event.set()