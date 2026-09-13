## main file ###

# --- imports ---
import os.path
import threading

import time
import re
import datetime

from simulation import *
from agent import *
from supervisor import *
from communicator import Communicator


# --- let's go ---
def main(session_dir, runs, tests):
    print("Starting")
    print("Open communication")
    bus = Communicator()
    ### ---- CHOOSE PARAMETERS HERE ---- ###
    ## agent ##
    agent = TheAgent(bus, runs, tests, session_dir)
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

    print("All threads closed cleanly. Exiting program.")


def get_success_rate(file_path: str) -> float:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"tests:\s*(\d+),\s*successes:\s*(\d+)", content)
    if not match:
        raise ValueError("Nie znaleziono wiersza ze statystykami testów.")

    tests = int(match.group(1))
    successes = int(match.group(2))

    if tests == 0:
        return 0.0

    return (successes / tests) * 100

import re

def get_runs(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"runs:\s*(\d+)", content)
    if not match:
        raise ValueError("Nie znaleziono informacji o liczbie uruchomień (runs).")

    return int(match.group(1))

def append_text_to_file(file_path: str, text: str, ensure_newline: bool = True) -> None:
    """
    Dopisuje podany tekst na koniec pliku.
    
    :param file_path: Ścieżka do pliku.
    :param text: Tekst do dopisania.
    :param ensure_newline: Jeśli True, upewnia się, że dopisywany tekst zaczyna się od nowej linii
                           oraz kończy znakiem nowej linii.
    """
    if ensure_newline:
        # Sprawdzamy, czy plik istnieje i czy kończy się już nową linią
        prefix = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content and not content.endswith("\n"):
                    prefix = "\n"
        except FileNotFoundError:
            pass

        # Upewniamy się, że dodawany tekst kończy się nową linią
        suffix = "" if text.endswith("\n") else "\n"
        text = prefix + text + suffix

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text)

def get_latest_session(logs_root):
    if not os.path.exists(logs_root): return None
    session_dirs = [os.path.join(logs_root, d) for d in os.listdir(logs_root) if os.path.isdir(os.path.join(logs_root, d))]
    return max(session_dirs, key=os.path.basename) if session_dirs else None

if __name__ == "__main__":
    ### Choose Parameters Here ###
    session_dir = 'continue'     # 'new' or 'continue' or <session dir> - direction where the brain is (under /logs/session_dir)
    runs = 1             # no. runs - either execution of plan or exploration - before test
    tests = 10              # how many times tries to go from start to finish from random spawn
    time_limit = 4.75       # in hours, breaks the loop
    min_success_rate = 0.7  # % of successful test to break the loop

    sum_runs = 0
    retries = 0
    start_time = time.time()
    while time.time() - start_time < time_limit*3600:
        retries += 1

        main(session_dir, runs, tests) # MAIN LOOP #

        session_dir = 'continue'
        stats_file_path = os.path.join(get_latest_session("logs"), "execution_statistics.txt")
        sum_runs += get_runs(stats_file_path)
        if get_success_rate(stats_file_path) >= min_success_rate:
            break

    duration = time.time() - start_time
    hours, remainder = divmod(int(duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    mssg = f"After {retries} attemps ({sum_runs} runs) program achieved success rate of {get_success_rate(stats_file_path):.2f}. Work time: {duration_str}\n"
    append_text_to_file(stats_file_path, f"")