# collect statistics from the robot
class StatsAgent():
    def __init__(self):
        pass
    def update(self):
        pass
    def close(self):
        pass

import networkx as nx
from datetime import datetime

class NetworkLogger(StatsAgent):
    """
    Logs the BrainNetwork's transitional map to a file periodically.
    """
    def __init__(self, agent, save_dir="brain_network", update_frequency=10):
        super().__init__()
        self.brain = agent.brain
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = f"logs/{start_time}/{save_dir}"
        self.live_path = f"{self.save_path}/transgraph_live.graphml"
        self.update_frequency = update_frequency
        self.step_counter = 0

    def update(self):
        self.step_counter += 1
        
        # Periodically save a "live" snapshot for the viewer script
        if self.step_counter % self.update_frequency == 0:
            if len(self.brain.transitional_map.nodes) > 0:
                try:
                    nx.write_graphml(self.brain.transitional_map, self.live_path)
                except Exception:
                    pass # Ignore temporary write collisions

    def close(self):
        """
        Saves the final raw NetworkX graph data.
        """
        print("\n[Statistics] Closing NetworkLogger... Saving final data.")
        final_path = f"{self.save_path}/transgraph_final.graphml"
        try:
            nx.write_graphml(self.brain.transitional_map, final_path)
            print(f"[Statistics] Final network structural data saved to: {final_path}")
        except Exception as e:
            print(f"[Statistics] Could not save GraphML: {e}")