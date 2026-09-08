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
    monitors = [LiveGraphMonitor(), LiveSensimotorMonitor(bus), LiveGNGMonitor()]
    print("[Supervisor] Loaded.")
    setup_complete_event.set() 
    
    try:
        while not stop_event.is_set():
            # updating statusboards and saving continuous data
            for monitor in monitors:
                monitor.update()
            time.sleep(time_step)
            pass

    except KeyboardInterrupt:
        print("[Supervisor] Thread stopped by user.")
    finally:
        for monitor in monitors:
            monitor.close()
        # pictures
        pics = PipelinePictureGenerator(logs_root="logs")
        pics.run()
        print("[Supervisor] Supervisor closed safely.")
