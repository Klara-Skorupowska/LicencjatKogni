import numpy as np
import networkx as nx

class BrainNetwork:
    def __init__(self):
        # 8 lidars + 9 colors + 4 edge counts = 21 dimensions
        self.state_dim = 21 
        
        # We use a dictionary so node IDs remain stable if nodes are deleted.
        # This is strictly required to keep the NetworkX map synchronized.
        self.gng_nodes = {
            0: np.random.rand(self.state_dim),
            1: np.random.rand(self.state_dim)
        }
        self.gng_edges = {}   # Format: {tuple(sorted(id1, id2)): age}
        self.gng_errors = {0: 0.0, 1: 0.0}
        
        # Hyperparameters
        self.epsilon_b = 0.2
        self.epsilon_n = 0.006
        self.max_age = 50
        self.lambda_gen = 10
        self.alpha = 0.5  # Error reduction factor
        self.spawn_threshold = 100   # Only spawn if the worst error is higher than this
        self.beta = 0.995            # Error decay factor per step
        self.step_counter = 0

        self.transitional_map = nx.DiGraph()
        self.transitional_map.add_node(0)
        self.transitional_map.add_node(1)

        # --- ENVIRONMENT BOUNDS TRACKERS ---
        # Start min at +infinity and max at -infinity
        self.env_min = np.full(self.state_dim, np.inf)
        self.env_max = np.full(self.state_dim, -np.inf)
        
        # Start weights at 1.0. They will update immediately on the first frame.
        self.weights = np.ones(self.state_dim)

    def _update_scaling(self, state: np.ndarray):
        """Updates global min/max bounds and recalculates weights."""
        # Expand bounds if the new state has values outside the known environment
        self.env_min = np.minimum(self.env_min, state)
        self.env_max = np.maximum(self.env_max, state)
        
        # Calculate the range (max - min) for each dimension
        value_range = self.env_max - self.env_min
        
        # CRITICAL: Prevent division by zero. If a sensor hasn't changed at all yet
        # (e.g., max and min are equal), default its range to 1.0.
        value_range[value_range == 0] = 1.0 
        
        # The weight is the inverse of the range.
        self.weights = 1.0 / value_range

    def _get_scaled_distance(self, state1, state2):
        """Helper to calculate distance without larger numbers destroying the math."""
        return np.linalg.norm((state1 - state2) * self.weights)

    def classify(self, state: np.ndarray) -> int:
        node_ids = list(self.gng_nodes.keys())
        node_matrix = np.array(list(self.gng_nodes.values()))
        
        # self.weights is now maintained dynamically!
        weighted_diff = (node_matrix - state) * self.weights
        distances = np.linalg.norm(weighted_diff, axis=1)
        
        bmu_index = int(np.argmin(distances))
        return node_ids[bmu_index]

    def update(self, prev_state: np.ndarray, skill_executed: object, current_state: np.ndarray):
        self.step_counter += 1

        self._update_scaling(prev_state)
        self._update_scaling(current_state)

        # ==========================================
        # 1. GROWING NEURAL GAS (State Space Update)
        # ==========================================
        
        node_ids = list(self.gng_nodes.keys())
        node_matrix = np.array(list(self.gng_nodes.values()))
        
        # Step A: Find BMU1 (closest) and BMU2 (second closest)
        weighted_diff = (node_matrix - current_state) * self.weights
        distances = np.linalg.norm(weighted_diff, axis=1)
        
        closest_indices = np.argsort(distances)
        bmu1_id = node_ids[closest_indices[0]]
        bmu2_id = node_ids[closest_indices[1]]
        bmu1_dist = distances[closest_indices[0]]

        # Step B: Increment edge ages connected to BMU1
        for edge in list(self.gng_edges.keys()):
            if bmu1_id in edge:
                self.gng_edges[edge] += 1

        # Step C: Add squared error to BMU1
        self.gng_errors[bmu1_id] += bmu1_dist ** 2
        for n in self.gng_errors:
            self.gng_errors[n] *= self.beta # slowly forget old errors

        # Step D: Move BMU1 and its topological neighbors closer to current state
        self.gng_nodes[bmu1_id] += self.epsilon_b * (current_state - self.gng_nodes[bmu1_id])
        
        neighbors = []
        for (u, v) in self.gng_edges.keys():
            if u == bmu1_id: neighbors.append(v)
            elif v == bmu1_id: neighbors.append(u)

        for n_id in neighbors:
            self.gng_nodes[n_id] += self.epsilon_n * (current_state - self.gng_nodes[n_id])

        # Step E: Create or reset the edge between BMU1 and BMU2
        new_edge = tuple(sorted((bmu1_id, bmu2_id)))
        self.gng_edges[new_edge] = 0

        # Step F: Prune old edges and isolated nodes
        edges_to_remove = [e for e, age in self.gng_edges.items() if age > self.max_age]
        for e in edges_to_remove:
            del self.gng_edges[e]

        # Find nodes that no longer have any edges
        active_nodes_with_edges = set()
        for u, v in self.gng_edges.keys():
            active_nodes_with_edges.update([u, v])
            
        nodes_to_remove = [n for n in self.gng_nodes.keys() if n not in active_nodes_with_edges]
        for n in nodes_to_remove:
            del self.gng_nodes[n]
            del self.gng_errors[n]
            # Also remove from NetworkX to keep them synced
            if self.transitional_map.has_node(n):
                self.transitional_map.remove_node(n)

        # Step G: Insert a new node every lambda_gen steps
        if self.step_counter % self.lambda_gen == 0 and len(self.gng_nodes) >= 2:
            # 1. Find node 'q' with maximum error
            q_id = max(self.gng_errors, key=self.gng_errors.get)
            
            # 2. Check if the network needs new node
            if self.gng_errors[q_id] > self.spawn_threshold:
                # 3. Find neighbor 'f' of 'q' with maximum error
                q_neighbors = [v for u, v in self.gng_edges.keys() if u == q_id] + \
                              [u for u, v in self.gng_edges.keys() if v == q_id]
            
                if q_neighbors:
                    f_id = max(q_neighbors, key=lambda n: self.gng_errors.get(n, 0))
                
                    # 4. Create new node 'r' halfway between 'q' and 'f'
                    new_id = max(self.gng_nodes.keys()) + 1
                    self.gng_nodes[new_id] = 0.5 * (self.gng_nodes[q_id] + self.gng_nodes[f_id])
                    self.transitional_map.add_node(new_id)
                
                    # 5. Update edges: remove (q, f), add (q, r) and (r, f)
                    old_edge = tuple(sorted((q_id, f_id)))
                    if old_edge in self.gng_edges:
                        del self.gng_edges[old_edge]
                    self.gng_edges[tuple(sorted((q_id, new_id)))] = 0
                    self.gng_edges[tuple(sorted((f_id, new_id)))] = 0
                
                    # 6. Decrease errors
                    self.gng_errors[q_id] *= self.alpha
                    self.gng_errors[f_id] *= self.alpha
                    self.gng_errors[new_id] = self.gng_errors[q_id]

        # ==========================================
        # 2. TRANSITIONAL MAP (Action Update)
        # ==========================================
        
        # We classify based on the states BEFORE the GNG updated its weights this frame, 
        # but classifying now is fine because topological shifts are tiny (epsilon_b = 0.2).
        prev_node_id = self.classify(prev_state)
        current_node_id = self.classify(current_state)

        # If the skill moved the robot between two different topological states
        if prev_node_id != current_node_id:
            if self.transitional_map.has_edge(prev_node_id, current_node_id):
                self.transitional_map[prev_node_id][current_node_id]['weight'] += 1
            else:
                self.transitional_map.add_edge(
                    prev_node_id, 
                    current_node_id, 
                    skill=skill_executed.__class__.__name__, 
                    weight=1
                )