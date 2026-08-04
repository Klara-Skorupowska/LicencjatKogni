## the growing neural gas and transitional map are here.

import numpy as np
import networkx as nx
from scipy.spatial import distance

class BrainNetwork:
    """
    Combines a Growing Neural Gas (GNG) for unsupervised state-space mapping 
    with a Transitional Map (Directed Graph) that records the skills required 
    to move between those states.
    """
    def __init__(self, state_dim: int, max_nodes: int = 1000):
        # --- 1. Growing Neural Gas (GNG) Properties ---
        # Instead of a heavy external library, pure Numpy is the fastest for PyBullet online learning.
        self.state_dim = state_dim
        self.max_nodes = max_nodes
        
        # Node positions in the state space (the "weights" of the GNG)
        # We start with two random nodes as per standard GNG initialization
        self.gng_nodes = np.random.rand(2, self.state_dim) 
        
        # GNG undirected edges tracking topological age: {(node1, node2): age}
        self.gng_edges = {}
        
        # GNG hyperparameters
        self.epsilon_b = 0.2     # Learning rate for Best Matching Unit (BMU)
        self.epsilon_n = 0.006   # Learning rate for neighbors
        self.max_age = 50        # Max age before an edge is removed
        self.lambda_gen = 100    # Steps between adding a new node
        self.step_counter = 0

        # --- 2. Transitional Map ---
        # A directed graph where Nodes = GNG Node IDs, Edges = Executed Skills
        self.transitional_map = nx.DiGraph()
        
        # Initialize the map with our first two GNG nodes
        self.transitional_map.add_node(0)
        self.transitional_map.add_node(1)

    def classify(self, state: np.ndarray) -> int:
        """
        Finds the Best Matching Unit (BMU) in the GNG for a given PyBullet state.
        Returns the integer ID of the closest node.
        """
        # Calculate Euclidean distance from the input state to all GNG nodes
        distances = distance.cdist([state], self.gng_nodes, metric='euclidean')[0]
        
        # Return the index of the closest node (BMU)
        bmu_index = np.argmin(distances)
        return int(bmu_index)

    def update(self, prev_state: np.ndarray, skill_executed: object, current_state: np.ndarray):
        """
        Updates both the GNG topology and the Transitional Map based on the robot's experience.
        
        prev_state: The PyBullet sensor array before the skill ran
        skill_executed: The Skill object (e.g., MoveForward instance) that was executed
        current_state: The PyBullet sensor array after the skill finished
        """
        self.step_counter += 1

        # 1. --- Update the GNG (State Space Mapping) ---
        # (Standard GNG algorithm steps go here)
        # - Find 1st and 2nd closest nodes (BMU1, BMU2) to current_state
        # - Increment age of all edges connected to BMU1
        # - Move BMU1 and its topological neighbors closer to current_state
        # - Create/reset edge between BMU1 and BMU2
        # - Remove edges older than max_age, and remove isolated nodes
        # - Every `lambda_gen` steps, insert a new node to reduce max error
        
        # ... GNG math implementation ...

        # 2. --- Update the Transitional Map (Action Mapping) ---
        # Map the continuous physics states to discrete graph nodes
        prev_node_id = self.classify(prev_state)
        current_node_id = self.classify(current_state)

        # If the skill successfully moved the robot to a new topological state, record it
        if prev_node_id != current_node_id:
            # Add or update the directed edge with the skill used to get there
            # We can also track success rates, execution counts, or average time taken here
            if self.transitional_map.has_edge(prev_node_id, current_node_id):
                # Update existing edge statistics (e.g., increment times traversed)
                self.transitional_map[prev_node_id][current_node_id]['weight'] += 1
            else:
                # Create new transition edge
                self.transitional_map.add_edge(
                    prev_node_id, 
                    current_node_id, 
                    skill=skill_executed.__class__.__name__, 
                    weight=1
                )
