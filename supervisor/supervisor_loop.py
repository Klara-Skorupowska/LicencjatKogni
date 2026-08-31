from communicator import Communicator
from .core import Supervisor
import threading
import time

def supervisor_loop(bus: Communicator, setup_complete_event, stop_event: threading.Event):
    """Runs continuously at 60Hz, completely independent of the agent and simulation."""
    print("[SUPERVISOR] Starting...")
    print("[SUPERVISOR] Loading statistics...")
    # status boards
    # colectiong data
    time_step = 1.0 / 60.0
    #world.setup(bus, time_step)
    # robot.set_end_pad_coords(arena.get_end_pad_coords()) useless
    setup_complete_event.set() 
    god = Supervisor(bus)
    print("[SUPERVISOR] Loaded.")
    
    try:
        while not stop_event.is_set():
            # updating statusboards and saving continuous data
            time.sleep(time_step)
            pass

    except KeyboardInterrupt:
        print("[SUPERVISOR] Thread stopped by user.")
    finally:
        #world.close()
        print("[SUPERVISOR] Supervisor closed safely.")
