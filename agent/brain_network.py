import os
import json
import numpy as np
import networkx as nx
from collections import defaultdict

class BrainNetwork:
    def __init__(self, log_dir=None):
        # We will dynamically set this upon receiving the first state vector
        self.state_dim = None 
        
        self.gng_nodes = {}
        self.gng_edges = {}
        self.gng_errors = {0: 0.0, 1: 0.0}
        
        # Hyperparameters
        self.epsilon_b = 0.2
        self.epsilon_n = 0.006
        self.max_age = 50
        self.alpha = 0.5
        self.spawn_threshold = 0.1 
        self.beta = 0.99 
        self.step_counter = 0

        self.transitional_map = nx.DiGraph()
        self.transitional_map.add_node(0)
        self.transitional_map.add_node(1)

        # Logging Setup
        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.symbols_dir = os.path.join(self.log_dir, "PDDL", "symbols")
            self.live_graphs_dir = os.path.join(self.log_dir, "graphs", "live")
            self.final_graphs_dir = os.path.join(self.log_dir, "graphs", "final")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.symbols_dir, exist_ok=True)
            os.makedirs(self.live_graphs_dir, exist_ok=True)
            os.makedirs(self.final_graphs_dir, exist_ok=True)

        self.env_min = None
        self.env_max = None
        self.weights = None


    def _update_scaling(self, state: np.ndarray):
        # Dynamically initialize dimensions if this is the first state read
        if self.state_dim is None:
            self.state_dim = len(state)
            self.gng_nodes = {
                0: np.random.rand(self.state_dim),
                1: np.random.rand(self.state_dim)
            }
            self.env_min = np.full(self.state_dim, np.inf)
            self.env_max = np.full(self.state_dim, -np.inf)
            self.weights = np.ones(self.state_dim)

        self.env_min = np.minimum(self.env_min, state)
        self.env_max = np.maximum(self.env_max, state)
        value_range = self.env_max - self.env_min
        value_range[value_range == 0] = 1.0 
        self.weights = 1.0 / value_range

    def _get_scaled_distance(self, state1, state2):
        return np.linalg.norm((state1 - state2) * self.weights)

    def _get_skill_name(self, skill_executed):
        if isinstance(skill_executed, str):
            return skill_executed
        return getattr(skill_executed, 'name', skill_executed.__class__.__name__)

    def classify(self, state: np.ndarray) -> int:
        node_ids = list(self.gng_nodes.keys())
        node_matrix = np.array(list(self.gng_nodes.values()))
        weighted_diff = (node_matrix - state) * self.weights
        distances = np.linalg.norm(weighted_diff, axis=1)
        bmu_index = int(np.argmin(distances))
        return node_ids[bmu_index]

    def update_gng(self, current_state: np.ndarray):
        print("[BRAIN] Updating GNG...")
        self._update_scaling(current_state)

        # ==========================================
        # 1. GROWING NEURAL GAS 
        # ==========================================
        node_ids = list(self.gng_nodes.keys())
        node_matrix = np.array(list(self.gng_nodes.values()))
        weighted_diff = (node_matrix - current_state) * self.weights
        distances = np.linalg.norm(weighted_diff, axis=1)
        closest_indices = np.argsort(distances)
        bmu1_id, bmu2_id = node_ids[closest_indices[0]], node_ids[closest_indices[1]]
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

    def update_tg(self, prev_state: np.ndarray, skill_executed, current_state: np.ndarray):
        print("[BRAIN] Updating Transitional Graph...")
        skill_name = self._get_skill_name(skill_executed)

        # ==========================================
        # 2. TRANSITIONAL MAP 
        # ==========================================
        prev_node_id = self.classify(prev_state)
        current_node_id = self.classify(current_state)

        if prev_node_id != current_node_id:
            if self.transitional_map.has_edge(prev_node_id, current_node_id):
                self.transitional_map[prev_node_id][current_node_id]['weight'] += 1
            else:
                self.transitional_map.add_edge(prev_node_id, current_node_id, skill=skill_name, weight=1)
        
        # new symbols from the updated transitional map
        self.create_symbols()
        # Continously write changes
        if self.log_dir:
            self.log_graphs(stage='live')

    def log_graphs(self, stage='live'):
        """Saves JSON for GNG and GraphML for Transitional Map."""
        if not self.log_dir: return
        target_dir = self.live_graphs_dir if stage == 'live' else self.final_graphs_dir
        
        # 1. Save Transitional Graph (Only connected nodes)
        connected_nodes = [n for n, d in self.transitional_map.degree() if d > 0]
        connected_subgraph = self.transitional_map.subgraph(connected_nodes).copy()
        
        # NetworkX requires string keys for boolean attributes in GraphML
        for n, data in connected_subgraph.nodes(data=True):
            if 'is_goal' in data:
                data['is_goal'] = str(data['is_goal'])

        graphml_path = os.path.join(target_dir, "transitional_graph.graphml")
        nx.write_graphml(connected_subgraph, graphml_path)

        # 2. Save GNG (JSON)
        gng_data = {
            "nodes": {str(k): v.tolist() for k, v in self.gng_nodes.items()},
            "edges": [{"source": u, "target": v} for u, v in self.gng_edges.keys()]
        }
        json_path = os.path.join(target_dir, "gng_graph.json")
        with open(json_path, 'w') as f:
            json.dump(gng_data, f, indent=4)


    def predict(self, state_id: int, skill_executed):
        if not self.transitional_map.has_node(state_id): return None
        skill_name = self._get_skill_name(skill_executed)
        for potential_next_state in self.transitional_map.successors(state_id):
            edge_data = self.transitional_map.get_edge_data(state_id, potential_next_state)
            if edge_data and edge_data.get('skill') == skill_name:
                return potential_next_state
        return None


    def create_symbols(self):

        print("[BRAIN] Creating symbols...")
        precondition_groups = defaultdict(list)
        effect_groups = defaultdict(list)
        
        for u, v, data in self.transitional_map.edges(data=True):
            skill = data.get('skill') 
            if skill:
                precondition_groups[skill].append(u)
                effect_groups[skill].append(v)
                
        symbolic_bounds = {'preconditions': {}, 'effects': {}}
        
        # 1. Process Preconditions and write .txt files
        for skill, raw_sources in precondition_groups.items():
            symbol_name = f"{skill}_precondition"
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
                
                # Directly output the .txt file
                if getattr(self, 'symbols_dir', None):
                    file_path = os.path.join(self.symbols_dir, f"{symbol_name}.txt")
                    with open(file_path, "w") as f:
                        f.write(f"MIN: {src_min.tolist()}\n")
                        f.write(f"MAX: {src_max.tolist()}\n")
                
        # 2. Process Effects and write .txt files
        for skill, raw_targets in effect_groups.items():
            symbol_name = f"{skill}_effect"
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
                
                # Directly output the .txt file
                if getattr(self, 'symbols_dir', None):
                    file_path = os.path.join(self.symbols_dir, f"{symbol_name}.txt")
                    with open(file_path, "w") as f:
                        f.write(f"MIN: {tgt_min.tolist()}\n")
                        f.write(f"MAX: {tgt_max.tolist()}\n")

    def get_symbolic_representation(self, state_vector=None):
        self.create_symbols()
        # Check satisfied symbols if a state vector was provided
        satisfied_symbols = []
        # use files created in create_symbols to check if the state_vector satisfies any precondition or effect
        if state_vector is not None and getattr(self, 'pddl_dir', None):
            symbols_dir = os.path.join(self.pddl_dir, "symbols")
            if os.path.exists(symbols_dir):
                for file in os.listdir(symbols_dir):
                    if file.endswith(".txt"):
                        symbol_name = file.replace(".txt", "")
                        file_path = os.path.join(symbols_dir, file)
                        with open(file_path, "r") as f:
                            lines = f.readlines()
                            min_line = next((line for line in lines if line.startswith("MIN:")), None)
                            max_line = next((line for line in lines if line.startswith("MAX:")), None)
                            if min_line and max_line:
                                min_values = np.array(eval(min_line.split("MIN:")[1].strip()))
                                max_values = np.array(eval(max_line.split("MAX:")[1].strip()))
                                if np.all(state_vector >= min_values) and np.all(state_vector <= max_values):
                                    satisfied_symbols.append(symbol_name)
            else:
                print(f"Symbols directory does not exist: {symbols_dir}")
                return None
        return satisfied_symbols

    def get_symbols(self):
        '''returns list of all symbols in files created in create_symbols'''
        self.create_symbols()  # Ensure symbols are up-to-date
        symbols = []
        if getattr(self, 'pddl_dir', None):
            symbols_dir = os.path.join(self.pddl_dir, "symbols")
            if os.path.exists(symbols_dir):
                for file in os.listdir(symbols_dir):
                    if file.endswith(".txt"):
                        symbols.append(file.replace(".txt", ""))
                return symbols
            else:
                print(f"Symbols directory does not exist: {symbols_dir}")
                return None
