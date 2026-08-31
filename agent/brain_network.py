import numpy as np
import networkx as nx

class BrainNetwork:
    def __init__(self):
        # 8 lidars + 9 colors + 4 edge counts = 21 dimensions
        self.state_dim = 21 
        
        # We use a dictionary so node IDs remain stable if nodes are deleted.
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
        self.alpha = 0.5            # Error reduction factor
        self.spawn_threshold = 0.1    # Only spawn if the worst error is higher than this
        self.beta = 0.99            # Error decay factor per step
        self.step_counter = 0

        self.transitional_map = nx.DiGraph()
        self.transitional_map.add_node(0)
        self.transitional_map.add_node(1)

        # --- ENVIRONMENT BOUNDS TRACKERS ---
        self.env_min = np.full(self.state_dim, np.inf)
        self.env_max = np.full(self.state_dim, -np.inf)
        self.weights = np.ones(self.state_dim)

    def _update_scaling(self, state: np.ndarray):
        """Updates global min/max bounds and recalculates weights."""
        self.env_min = np.minimum(self.env_min, state)
        self.env_max = np.maximum(self.env_max, state)
        value_range = self.env_max - self.env_min
        value_range[value_range == 0] = 1.0 
        self.weights = 1.0 / value_range

    def _get_scaled_distance(self, state1, state2):
        """Helper to calculate distance without larger numbers destroying the math."""
        return np.linalg.norm((state1 - state2) * self.weights)

    def _get_skill_name(self, skill_executed):
        """
        Helper to robustly extract the name of a skill.
        Ensures PDDL strings match the agent's skillset dict keys (e.g., 'SpotTheDoor' vs 'SpotTheColor').
        """
        if isinstance(skill_executed, str):
            return skill_executed
        # If an object is passed, check if it has a defined '.name' property, otherwise fallback to class name
        return getattr(skill_executed, 'name', skill_executed.__class__.__name__)

    def classify(self, state: np.ndarray) -> int:
        node_ids = list(self.gng_nodes.keys())
        node_matrix = np.array(list(self.gng_nodes.values()))
        
        weighted_diff = (node_matrix - state) * self.weights
        distances = np.linalg.norm(weighted_diff, axis=1)
        
        bmu_index = int(np.argmin(distances))
        return node_ids[bmu_index]

    def update(self, prev_state: np.ndarray, skill_executed, current_state: np.ndarray):
        '''
        Updates both networks. Accessed only when prediction fails (new environment state).
        '''
        print("[BRAIN] Updating networks...")
        self._update_scaling(prev_state)
        self._update_scaling(current_state)

        # Extract the exact string name for the transition map edge
        skill_name = self._get_skill_name(skill_executed)

        # ==========================================
        # 1. GROWING NEURAL GAS (State Space Update)
        # ==========================================
        
        node_ids = list(self.gng_nodes.keys())
        node_matrix = np.array(list(self.gng_nodes.values()))
        
        weighted_diff = (node_matrix - current_state) * self.weights
        distances = np.linalg.norm(weighted_diff, axis=1)
        
        closest_indices = np.argsort(distances)
        bmu1_id = node_ids[closest_indices[0]]
        bmu2_id = node_ids[closest_indices[1]]
        bmu1_dist = distances[closest_indices[0]]

        for edge in list(self.gng_edges.keys()):
            if bmu1_id in edge:
                self.gng_edges[edge] += 1

        self.gng_errors[bmu1_id] += bmu1_dist ** 2
        for n in self.gng_errors:
            self.gng_errors[n] *= self.beta

        self.gng_nodes[bmu1_id] += self.epsilon_b * (current_state - self.gng_nodes[bmu1_id])
        
        neighbors = []
        for (u, v) in self.gng_edges.keys():
            if u == bmu1_id: neighbors.append(v)
            elif v == bmu1_id: neighbors.append(u)

        for n_id in neighbors:
            self.gng_nodes[n_id] += self.epsilon_n * (current_state - self.gng_nodes[n_id])

        new_edge = tuple(sorted((bmu1_id, bmu2_id)))
        self.gng_edges[new_edge] = 0

        edges_to_remove = [e for e, age in self.gng_edges.items() if age > self.max_age]
        for e in edges_to_remove:
            del self.gng_edges[e]

        active_nodes_with_edges = set()
        for u, v in self.gng_edges.keys():
            active_nodes_with_edges.update([u, v])
            
        nodes_to_remove = [n for n in self.gng_nodes.keys() if n not in active_nodes_with_edges]
        for n in nodes_to_remove:
            del self.gng_nodes[n]
            del self.gng_errors[n]
            if self.transitional_map.has_node(n):
                self.transitional_map.remove_node(n)

        if len(self.gng_nodes) >= 2:
            q_id = max(self.gng_errors, key=self.gng_errors.get)
            
            if self.gng_errors[q_id] > self.spawn_threshold:
                q_neighbors = [v for u, v in self.gng_edges.keys() if u == q_id] + \
                              [u for u, v in self.gng_edges.keys() if v == q_id]
            
                if q_neighbors:
                    f_id = max(q_neighbors, key=lambda n: self.gng_errors.get(n, 0))
                
                    new_id = max(self.gng_nodes.keys()) + 1
                    self.gng_nodes[new_id] = 0.5 * (self.gng_nodes[q_id] + self.gng_nodes[f_id])
                    self.transitional_map.add_node(new_id)
                
                    old_edge = tuple(sorted((q_id, f_id)))
                    if old_edge in self.gng_edges:
                        del self.gng_edges[old_edge]
                    self.gng_edges[tuple(sorted((q_id, new_id)))] = 0
                    self.gng_edges[tuple(sorted((f_id, new_id)))] = 0
                
                    self.gng_errors[q_id] *= self.alpha
                    self.gng_errors[f_id] *= self.alpha
                    self.gng_errors[new_id] = self.gng_errors[q_id]

        # ==========================================
        # 2. TRANSITIONAL MAP (Action Update)
        # ==========================================
        
        prev_node_id = self.classify(prev_state)
        current_node_id = self.classify(current_state)

        if prev_node_id != current_node_id:
            if self.transitional_map.has_edge(prev_node_id, current_node_id):
                self.transitional_map[prev_node_id][current_node_id]['weight'] += 1
            else:
                self.transitional_map.add_edge(
                    prev_node_id, 
                    current_node_id, 
                    skill=skill_name,  # Tagged with string instead of object reference
                    weight=1
                )

    def predict(self, state_id: int, skill_executed):
        '''
        Predicts the next state ID by checking the transitional map.
        '''
        if not self.transitional_map.has_node(state_id):
            return None
            
        skill_name = self._get_skill_name(skill_executed)
        
        for potential_next_state in self.transitional_map.successors(state_id):
            edge_data = self.transitional_map.get_edge_data(state_id, potential_next_state)
            
            if edge_data and edge_data.get('skill') == skill_name:
                return potential_next_state
                
        return None

    def get_symbolic_representation(self):
        '''
        Dynamically aggregates the raw GNG and transitional map into purely skill-based Preconditions and Effects.
        '''
        from collections import defaultdict
        
        precondition_groups = defaultdict(list)
        effect_groups = defaultdict(list)
        
        # 1. Map continuous nodes based solely on the skill string mapped to the edge
        for u, v, data in self.transitional_map.edges(data=True):
            skill = data.get('node')
            if skill:
                precondition_groups[skill].append(u)
                effect_groups[skill].append(v)
                
        symbolic_bounds = {
            'preconditions': {},
            'effects': {}
        }
        
        # 2. Extract bounding boxes for Preconditions (pre_<skill>)
        for skill, raw_sources in precondition_groups.items():
            symbol_name = f"pre_{skill}"
            unique_sources = list(set(raw_sources))
            
            source_vectors = [self.gng_nodes[n] for n in unique_sources if n in self.gng_nodes]
            
            if source_vectors:
                src_min = np.min(source_vectors, axis=0)
                src_max = np.max(source_vectors, axis=0)
                symbolic_bounds['preconditions'][symbol_name] = {
                    'min': src_min,
                    'max': src_max,
                    'raw_nodes': unique_sources,
                    'skill': skill
                }
                
        # 3. Extract bounding boxes for Effects (eff_<skill>)
        for skill, raw_targets in effect_groups.items():
            symbol_name = f"eff_{skill}"
            unique_targets = list(set(raw_targets))
            
            target_vectors = [self.gng_nodes[n] for n in unique_targets if n in self.gng_nodes]
            
            if target_vectors:
                tgt_min = np.min(target_vectors, axis=0)
                tgt_max = np.max(target_vectors, axis=0)
                symbolic_bounds['effects'][symbol_name] = {
                    'min': tgt_min,
                    'max': tgt_max,
                    'raw_nodes': unique_targets,
                    'skill': skill
                }
                
        return symbolic_bounds