from communicator import Communicator
from .core import Supervisor
import threading
import time
from .visualize import *

def supervisor_loop(bus: Communicator, setup_complete_event, stop_event: threading.Event):
    """Runs continuously at 5Hz, completely independent of the agent and simulation."""
    print("[Supervisor] Starting...")
    print("[Supervisor] Loading statistics...")
    # status boards
    # colectiong data
    time_step = 1.0 / 5.0
    god = Supervisor(bus)
    god.setup()
    monitor = LiveGraphMonitor()
    senses = LiveSensimotorMonitor(bus)
    print("[Supervisor] Loaded.")
    setup_complete_event.set() 
    
    try:
        while not stop_event.is_set():
            # updating statusboards and saving continuous data
            monitor.update()
            senses.update()
            time.sleep(time_step)
            pass

    except KeyboardInterrupt:
        print("[Supervisor] Thread stopped by user.")
    finally:
        monitor.close()
        senses.close()
        # pictures
        runner = PipelineVisualizer(logs_root="logs")
        runner.run_all()
        print("[Supervisor] Supervisor closed safely.")
