import os
import json
import numpy as np
import networkx as nx
from collections import defaultdict

class BrainNetwork:
    def __init__(self, log_dir=None):
        self.state_dim = None 
        
        # GNG-U Data Structures
        self.gng_nodes = {}
        self.gng_edges = {}
        self.gng_errors = {}
        self.gng_utility = {}         # Utility tracker for each prototype
        self.node_visit_counts = {}   # Tracks visits to decay learning rate per prototype
        
        # Hyperparameters
        self.epsilon_b = 0.2          # Initial BMU movement rate
        self.epsilon_n = 0.006        # Initial neighbor movement rate
        self.eps_decay = 0.005        # Prototype-level decay rate
        self.eps_min = 0.001          # Floor rate to retain slight plastic adaptability
        self.max_age = 50             # Maximum edge age before removal
        self.alpha = 0.5              # Error reduction factor on split
        self.beta = 0.99              # Global error & utility decay per step
        self.lambda_step = 1          # Spawn/utility replacement evaluation interval
        self.max_nodes = 100          # Upper bound on network capacity
        self.step_counter = 0         # Step tracker for lambda intervals

        # MultiDiGraph begins completely empty; populated strictly via update_tg
        self.transitional_map = nx.MultiDiGraph()
        self.node_mapping = {}
        self.symbols = {}

        # Logging Setup
        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.symbols_dir = os.path.join(self.log_dir, "PDDL", "symbols")
            self.graphs_dir = os.path.join(self.log_dir, "graphs")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.symbols_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)

        self.env_min = None
        self.env_max = None
        self.weights = None

    def _update_scaling(self, state: np.ndarray):
        if self.state_dim is None:
            self.state_dim = len(state)
            self.gng_nodes = {
                0: np.random.rand(self.state_dim),
                1: np.random.rand(self.state_dim)
            }
            self.gng_edges = {(0, 1): 0}  # Ensure initial nodes are interconnected
            self.gng_errors = {0: 0.0, 1: 0.0}
            self.gng_utility = {0: 1.0, 1: 1.0}
            self.node_visit_counts = {0: 0, 1: 0}
            self.env_min = np.full(self.state_dim, np.inf)
            self.env_max = np.full(self.state_dim, -np.inf)
            self.weights = np.ones(self.state_dim)

        self.env_min = np.minimum(self.env_min, state)
        self.env_max = np.maximum(self.env_max, state)
        
        # Calculate the actual raw range
        value_range = self.env_max - self.env_min
        
        # Initialize all weights to 0.0 (masking out static dimensions)
        self.weights = np.zeros_like(value_range)
        
        # Apply standard scaling ONLY to dimensions with meaningful variance
        active_dims = value_range > 1e-4
        self.weights[active_dims] = 1.0 / value_range[active_dims]

    def _get_scaled_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Computes weighted Euclidean distance using environment range scaling."""
        if self.weights is None:
            return float(np.linalg.norm(state1 - state2))
        return float(np.linalg.norm((state1 - state2) * self.weights))

    def _get_merged_node(self, node_id):
        """Resolves an alias or original node ID to the active transitional graph node."""
        # Check direct containment (string and int)
        for candidate in (node_id, str(node_id)):
            if self.transitional_map.has_node(candidate):
                return candidate
            if candidate in self.node_mapping:
                target = self.node_mapping[candidate]
                if self.transitional_map.has_node(target):
                    return target

        # Check aliases metadata
        for n, data in self.transitional_map.nodes(data=True):
            aliases = data.get('aliases', [])
            if node_id in aliases or str(node_id) in [str(a) for a in aliases]:
                self.node_mapping[node_id] = n
                self.node_mapping[str(node_id)] = n
                return n

        return str(node_id)

    def classify(self, state: np.ndarray) -> int:
        self._update_scaling(state)
        node_ids = list(self.gng_nodes.keys())
        node_matrix = np.array(list(self.gng_nodes.values()))
        weighted_diff = (node_matrix - state) * self.weights
        distances = np.linalg.norm(weighted_diff, axis=1)
        bmu_index = int(np.argmin(distances))
        return node_ids[bmu_index]

    def predict(self, state: np.ndarray, skill_name: str):
        """
        Predicts the resulting grounded predicate tuple ("at", "node_Y") by applying 
        the action semantics declared in domain.pddl against the current problem.pddl.
        """
        domain_file = os.path.join(self.pddl_dir, "domain.pddl")
        problem_file = os.path.join(self.pddl_dir, "problem.pddl")

        if not os.path.exists(domain_file) or not os.path.exists(problem_file):
            return None

        # 1. Resolve current state object from continuous vector
        resolved_preds = self.resolve_predicates(state)
        current_loc_obj = next((args[0] for pred, *args in resolved_preds if pred == "at"), None)
        if not current_loc_obj:
            bmu = self.classify(state)
            merged = self.get_merged_node(bmu)
            current_loc_obj = f"node_{merged}"

        # 2. Extract allowed grounded transitions from problem.pddl (:init ...)
        # Checks for: (can_<skill_name> <current_loc_obj> <target_obj>)
        expected_fact_prefix = f"can_{skill_name.lower()} {current_loc_obj.lower()}"
        target_obj = None

        with open(problem_file, "r") as f:
            for line in f:
                line = line.strip().lower()
                if not line or line.startswith(";"):
                    continue
                fact = line.strip("()")
                if fact.startswith(expected_fact_prefix):
                    tokens = fact.split()
                    if len(tokens) == 3:
                        target_obj = tokens[2]
                        break

        if not target_obj:
            return None

        # 3. Verify in domain.pddl that this action produces the effect (at ?to)
        # Finds the action block matching skill_name and ensures (at ?to) is in :effect
        action_found = False
        effect_verified = False
        with open(domain_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                cleaned = line.strip().lower()
                if cleaned.startswith(f"(:action {skill_name.lower()}"):
                    action_found = True
                if action_found:
                    if ":effect" in cleaned:
                        effect_snippet = "".join(lines[i:i+4]).lower()
                        if "(at ?to)" in effect_snippet:
                            effect_verified = True
                        break

        if effect_verified:
            # Preserve matching casing with self.symbols if available
            matched_name = next((k for k in self.symbols.keys() if k.lower() == target_obj), target_obj)
            return ("at", matched_name)

        return None

    def update_gng(self, current_state: np.ndarray):
        print("\t[BRAIN] Updating GNG...")
        self.step_counter += 1
        self._update_scaling(current_state)

        node_ids = list(self.gng_nodes.keys())
        node_matrix = np.array(list(self.gng_nodes.values()))
        weighted_diff = (node_matrix - current_state) * self.weights
        distances = np.linalg.norm(weighted_diff, axis=1)
        closest_indices = np.argsort(distances)
        bmu1_id, bmu2_id = node_ids[closest_indices[0]], node_ids[closest_indices[1]]
        bmu1_dist = distances[closest_indices[0]]
        bmu2_dist = distances[closest_indices[1]]

        print(f"\t\t[DEBUG GNG] BMU1: {bmu1_id} (dist: {bmu1_dist:.4f}), BMU2: {bmu2_id}")

        # 1. Edge aging for BMU1
        for edge in list(self.gng_edges.keys()):
            if bmu1_id in edge:
                self.gng_edges[edge] += 1

        # 2. Accumulate squared error & GNG-U Utility (utility = delta error = error2 - error1)
        self.gng_errors[bmu1_id] += bmu1_dist ** 2
        delta_err = (bmu2_dist ** 2) - (bmu1_dist ** 2)
        self.gng_utility[bmu1_id] = self.gng_utility.get(bmu1_id, 0.0) + delta_err

        # Decay all errors and utilities
        for n in self.gng_errors:
            self.gng_errors[n] *= self.beta
            if n in self.gng_utility:
                self.gng_utility[n] *= self.beta

        # 3. Dynamic Learning Rate Decay per prototype (stabilizes Voronoi partitions)
        self.node_visit_counts[bmu1_id] = self.node_visit_counts.get(bmu1_id, 0) + 1
        bmu_visits = self.node_visit_counts[bmu1_id]
        effective_eps_b = max(self.eps_min, self.epsilon_b / (1.0 + self.eps_decay * bmu_visits))
        effective_eps_n = max(self.eps_min, self.epsilon_n / (1.0 + self.eps_decay * bmu_visits))

        # Shift BMU1 and its neighbors
        self.gng_nodes[bmu1_id] += effective_eps_b * (current_state - self.gng_nodes[bmu1_id])
        
        neighbors = [v for u, v in self.gng_edges.keys() if u == bmu1_id] + \
                    [u for u, v in self.gng_edges.keys() if v == bmu1_id]

        for n_id in neighbors:
            self.gng_nodes[n_id] += effective_eps_n * (current_state - self.gng_nodes[n_id])

        # 4. Connect BMU1 and BMU2 with a fresh edge
        new_edge = tuple(sorted((bmu1_id, bmu2_id)))
        self.gng_edges[new_edge] = 0

        # 5. Age out old edges
        edges_to_remove = [e for e, age in self.gng_edges.items() if age > self.max_age]
        for e in edges_to_remove:
            del self.gng_edges[e]

        if edges_to_remove:
            print(f"\t\t[DEBUG GNG] Aged out {len(edges_to_remove)} edges older than max_age={self.max_age}.")

        # 6. Prune isolated nodes (safely decoupled from TG removal)
        active_nodes_with_edges = set()
        for u, v in self.gng_edges.keys():
            active_nodes_with_edges.update([u, v])
            
        nodes_to_remove = [n for n in self.gng_nodes.keys() if n not in active_nodes_with_edges]
        for n in nodes_to_remove:
            del self.gng_nodes[n]
            self.gng_errors.pop(n, None)
            self.gng_utility.pop(n, None)
            self.node_visit_counts.pop(n, None)

            # Reassign aliases rather than dropping TG nodes to prevent plan corruption
            tg_node = self._get_merged_node(n)
            surviving = list(self.gng_nodes.keys())
            if surviving and self.transitional_map.has_node(tg_node):
                closest = min(surviving, key=lambda s: np.linalg.norm((self.gng_nodes[s] - current_state) * self.weights))
                new_tg = self._get_merged_node(closest)
                self.node_mapping[n] = new_tg

        if nodes_to_remove:
            print(f"\t\t[DEBUG GNG] Pruned {len(nodes_to_remove)} isolated nodes: {nodes_to_remove}")

        # 7. GNG-U Node Insertion / Low-Utility Replacement
        if (self.step_counter % self.lambda_step == 0) and len(self.gng_errors) >= 2:
            # Check if capacity reached: remove lowest utility node if saturated
            if len(self.gng_nodes) >= self.max_nodes:
                q_max_err = max(self.gng_errors.values())
                low_u_id = min(self.gng_utility, key=self.gng_utility.get)

                # If removing the low-utility node does not severely hurt error representation
                if self.gng_utility[low_u_id] < (0.1 * q_max_err):
                    print(f"\t\t[DEBUG GNG-U] Capacity reached ({self.max_nodes}). Replacing low utility node {low_u_id} (U={self.gng_utility[low_u_id]:.4f}).")
                    # Delete edges connected to low-utility node
                    for edge in list(self.gng_edges.keys()):
                        if low_u_id in edge:
                            del self.gng_edges[edge]
                    del self.gng_nodes[low_u_id]
                    self.gng_errors.pop(low_u_id, None)
                    self.gng_utility.pop(low_u_id, None)
                    self.node_visit_counts.pop(low_u_id, None)

            # Insert node at highest error point if below capacity
            if len(self.gng_nodes) < self.max_nodes and len(self.gng_errors) >= 2:
                q_id = max(self.gng_errors, key=self.gng_errors.get)
                q_neighbors = [v for u, v in self.gng_edges.keys() if u == q_id] + \
                              [u for u, v in self.gng_edges.keys() if v == q_id]
                
                if q_neighbors:
                    f_id = max(q_neighbors, key=lambda n: self.gng_errors.get(n, 0))
                    new_id = max(self.gng_nodes.keys()) + 1
                    
                    # Split vector
                    self.gng_nodes[new_id] = 0.5 * (self.gng_nodes[q_id] + self.gng_nodes[f_id])

                    # Rewire topology
                    old_edge = tuple(sorted((q_id, f_id)))
                    if old_edge in self.gng_edges:
                        del self.gng_edges[old_edge]
                    self.gng_edges[tuple(sorted((q_id, new_id)))] = 0
                    self.gng_edges[tuple(sorted((f_id, new_id)))] = 0
                
                    # Redistribute errors and utilities
                    self.gng_errors[q_id] *= self.alpha
                    self.gng_errors[f_id] *= self.alpha
                    self.gng_errors[new_id] = self.gng_errors[q_id]

                    self.gng_utility[new_id] = 0.5 * (self.gng_utility.get(q_id, 1.0) + self.gng_utility.get(f_id, 1.0))
                    self.node_visit_counts[new_id] = 0

                    print(f"\t\t[DEBUG GNG-U] (Step {self.step_counter}) Spawned node {new_id} between {q_id} and {f_id}. Total nodes: {len(self.gng_nodes)}")

    def update_tg(self, prev_state: np.ndarray, skill_name, current_state: np.ndarray):
        print("\t[BRAIN] Updating Transitional Graph...")

        prev_gng = self.classify(prev_state)
        curr_gng = self.classify(current_state)

        prev_node_id = self._get_merged_node(prev_gng)
        current_node_id = self._get_merged_node(curr_gng)

        print(f"\t\t[DEBUG TG CLASSIFY] Raw BMUs: {prev_gng} ({prev_node_id}) -> {curr_gng} ({current_node_id}) via {skill_name}")

        # Register nodes dynamically only upon an observed transition
        if not self.transitional_map.has_node(prev_node_id):
            self.transitional_map.add_node(prev_node_id, aliases=[prev_gng])
            self.node_mapping[prev_gng] = prev_node_id

        if not self.transitional_map.has_node(current_node_id):
            self.transitional_map.add_node(current_node_id, aliases=[curr_gng])
            self.node_mapping[curr_gng] = current_node_id

        found_edge_key = None
        if self.transitional_map.has_edge(prev_node_id, current_node_id):
            for key, edge_data in self.transitional_map[prev_node_id][current_node_id].items():
                if edge_data.get('skill') == skill_name:
                    found_edge_key = key
                    break

        if found_edge_key is not None:
            self.transitional_map[prev_node_id][current_node_id][found_edge_key]['weight'] += 1
            w = self.transitional_map[prev_node_id][current_node_id][found_edge_key]['weight']
        else:
            self.transitional_map.add_edge(
                prev_node_id,
                current_node_id,
                key=skill_name,
                skill=skill_name,
                weight=1
            )
            w = 1

        print(f"\t\t[DEBUG TG] Edge: ({prev_node_id}) --[{skill_name} | weight={w}]--> ({current_node_id})")

        self._agregate_tg()
        if self.log_dir:
            self._log_graphs()

    def _agregate_tg(self):
        """Aggregates transitional graph based strictly on co-occurring skills and connected components."""
        print("\t[BRAIN] Aggregating Transitional Graph...")
        init_n = self.transitional_map.number_of_nodes()
        init_e = self.transitional_map.number_of_edges()

        # 1. Group targets by (source, skill), excluding self-loops to prevent origin-collapse
        merge_candidates = defaultdict(set)
        for u, v, data in self.transitional_map.edges(data=True):
            skill = data.get('skill')
            if u != v:  # Never merge external destinations with the origin node
                merge_candidates[(u, skill)].add(v)

        # 2. Build equivalence classes over ALL nodes so isolated and goal nodes are preserved
        merge_graph = nx.Graph()
        for node in self.transitional_map.nodes():
            merge_graph.add_node(node)

        for (u, skill), targets in merge_candidates.items():
            if len(targets) > 1:
                target_list = list(targets)
                first = target_list[0]
                for other in target_list[1:]:
                    merge_graph.add_edge(first, other)

        # 3. Create node mappings, aliases, and inherit metadata (such as is_goal)
        new_node_mapping = {}
        new_node_data = {}

        for component in nx.connected_components(merge_graph):
            comp_list = list(component)
            all_aliases = []
            is_goal_node = False

            for n in comp_list:
                node_data = self.transitional_map.nodes[n]
                if node_data.get('is_goal') in [True, 'True']:
                    is_goal_node = True

                aliases = node_data.get('aliases')
                if aliases:
                    for a in aliases:
                        if isinstance(a, int):
                            all_aliases.append(a)
                        elif isinstance(a, str):
                            all_aliases.extend([int(x) for x in a.split("_") if x.isdigit()])
                elif isinstance(n, int):
                    all_aliases.append(n)
                elif isinstance(n, str):
                    all_aliases.extend([int(x) for x in n.split("_") if x.isdigit()])

            sorted_aliases = sorted(set(all_aliases))
            new_node_name = "_".join(map(str, sorted_aliases)) if (len(comp_list) > 1 or len(sorted_aliases) > 1) else str(comp_list[0])

            new_node_data[new_node_name] = {
                'aliases': sorted_aliases if sorted_aliases else [comp_list[0]],
                'is_goal': is_goal_node
            }
            for old_n in comp_list:
                new_node_mapping[old_n] = new_node_name

        # 4. Construct aggregated MultiDiGraph
        new_tg = nx.MultiDiGraph()
        for new_node, data in new_node_data.items():
            new_tg.add_node(new_node, aliases=data['aliases'], is_goal=data['is_goal'])

        edge_data_collector = defaultdict(int)
        for u, v, data in self.transitional_map.edges(data=True):
            if u in new_node_mapping and v in new_node_mapping:
                new_u = new_node_mapping[u]
                new_v = new_node_mapping[v]
                skill = data.get('skill')
                weight = data.get('weight', 1)
                edge_data_collector[(new_u, new_v, skill)] += weight

        for (u, v, skill), total_weight in edge_data_collector.items():
            new_tg.add_edge(u, v, key=skill, skill=skill, weight=total_weight)

        # 5. Dynamically replace TG and update alias table
        self.transitional_map = new_tg
        self.node_mapping = {}
        for target_node, data in new_node_data.items():
            for alias in data['aliases']:
                self.node_mapping[alias] = target_node

        print(f"\t\t[DEBUG TG] Aggregated: {init_n} -> {new_tg.number_of_nodes()} nodes, {init_e} -> {new_tg.number_of_edges()} edges")
        self.create_symbols()

    def _log_graphs(self):
        """Saves JSON for GNG and GraphML for Transitional MultiGraph."""
        if not self.log_dir: return
        target_dir = self.graphs_dir
        
        # Include active transition nodes as well as designated goal nodes
        graph_to_log = self.transitional_map.copy()
        nodes_to_keep = [
            n for n, d in graph_to_log.degree() 
            if d > 0 or graph_to_log.nodes[n].get('is_goal') in [True, 'True']
        ]
        subgraph = graph_to_log.subgraph(nodes_to_keep).copy() if nodes_to_keep else graph_to_log.copy()
        
        for _, data in subgraph.nodes(data=True):
            if 'aliases' in data:
                data['aliases'] = str(data['aliases'])
            if 'is_goal' in data:
                data['is_goal'] = str(data['is_goal'])

        graphml_path = os.path.join(target_dir, "transitional_graph.graphml")
        nx.write_graphml(subgraph, graphml_path)

        gng_data = {
            "nodes": {str(k): v.tolist() for k, v in self.gng_nodes.items()},
            "edges": [{"source": u, "target": v} for u, v in self.gng_edges.keys()]
        }
        json_path = os.path.join(target_dir, "gng_graph.json")
        with open(json_path, 'w') as f:
            json.dump(gng_data, f, indent=4)

    def create_symbols(self):
        """
        Builds grounded symbol specs (centroid, radius, bounding intervals)
        exclusively for active transitional graph nodes and persists them to disk.
        """
        print("\t[BRAIN] Create symbols")
        self.symbols = {}

        # Purge stale symbol text files
        if getattr(self, 'symbols_dir', None) and os.path.exists(self.symbols_dir):
            for old_file in os.listdir(self.symbols_dir):
                if old_file.endswith(".txt"):
                    try:
                        os.remove(os.path.join(self.symbols_dir, old_file))
                    except OSError:
                        pass

        for node, data in self.transitional_map.nodes(data=True):
            # Only generate grounded symbols for nodes with active transitions
            if self.transitional_map.degree(node) == 0:
                continue

            aliases = data.get('aliases', [node])
            vectors = [self.gng_nodes[a] for a in aliases if a in self.gng_nodes]

            if vectors:
                min_bounds = np.min(vectors, axis=0)
                max_bounds = np.max(vectors, axis=0)
                centroid = np.mean(vectors, axis=0)

                if len(vectors) > 1:
                    dists = [self._get_scaled_distance(centroid, v) for v in vectors]
                    radius = float(max(dists) * 1.2)
                else:
                    radius = 0.25

                obj_name = f"node_{node}"
                self.symbols[obj_name] = {
                    'node': node,
                    'min': min_bounds,
                    'max': max_bounds,
                    'centroid': centroid,
                    'radius': radius,
                    'aliases': aliases
                }

                # Save symbol bounds for SensimotorVisualizer
                if getattr(self, 'symbols_dir', None):
                    file_path = os.path.join(self.symbols_dir, f"{obj_name}.txt")
                    with open(file_path, "w") as f:
                        f.write(f"PREDICATE: (at {obj_name})\n")
                        f.write(f"ALIASES: {aliases}\n")
                        f.write(f"RADIUS: {radius:.4f}\n")
                        f.write(f"MIN: {min_bounds.tolist()}\n")
                        f.write(f"MAX: {max_bounds.tolist()}\n")

    def resolve_predicates(self, state_vector: np.ndarray) -> list:
        """
        Resolves a continuous state vector into active PDDL atomic propositions:
        Returns list of tuples, e.g. [('at', 'node_0')].
        """
        if not self.symbols:
            self.create_symbols()

        if state_vector is None or len(self.gng_nodes) == 0:
            return []

        # 1. Classify state into GNG BMU and map to TG node
        bmu_id = self.classify(state_vector)
        tg_node = self._get_merged_node(bmu_id)
        obj_name = f"node_{tg_node}"

        # 2. Check if state lies within the topological symbol's radius
        if obj_name in self.symbols:
            spec = self.symbols[obj_name]
            dist = self._get_scaled_distance(state_vector, spec['centroid'])
            if dist <= spec['radius']:
                return [("at", obj_name)]

        # Fallback: exact Voronoi cell membership
        return [("at", obj_name)]

    def generate_domain_pddl(self, domain_name="continuous_world") -> str:
        """
        Generates domain.pddl where action names match skill names exactly,
        parameterized by source and destination nodes: (?from ?to).
        """
        print("\t[BRAIN] Generate domain.pddl")

        # Collect distinct skills with transitions in the graph
        skills = sorted({
            data.get('skill') for _, _, data in self.transitional_map.edges(data=True)
            if data.get('skill')
        })

        pddl = [f"(define (domain {domain_name})"]
        pddl.append("  (:requirements :strips :negative-preconditions)")

        # Predicates: agent location and skill-specific transition validity
        pddl.append("  (:predicates")
        pddl.append("    (at ?loc)")
        for skill in skills:
            pddl.append(f"    (can_{skill} ?from ?to)")
        pddl.append("  )\n")

        # Action definition: action == skill name
        for skill in skills:
            pddl.append(f"  (:action {skill}")
            pddl.append("    :parameters (?from ?to)")
            pddl.append(f"    :precondition (and (at ?from) (can_{skill} ?from ?to))")
            pddl.append("    :effect (and (at ?to) (not (at ?from)))")
            pddl.append("  )\n")

        pddl.append(")")
        return "\n".join(pddl)

    def generate_problem_pddl(self, initial_state, goal_state, 
                              problem_name="plan_problem", domain_name="continuous_world") -> str:
        """
        Generates problem.pddl declaring nodes as objects and active transitional
        edges as (can_<skill> <from> <to>) init facts.
        """
        print("\t[BRAIN] Generate problem.pddl")

        init_node = self._get_merged_node(
            self.classify(initial_state) if isinstance(initial_state, np.ndarray) else initial_state
        )
        goal_node = self._get_merged_node(
            self.classify(goal_state) if isinstance(goal_state, np.ndarray) else goal_state
        )

        def _fmt(n):
            s = str(n)
            return s if s.startswith("node_") else f"node_{s}"

        init_obj = _fmt(init_node)
        goal_obj = _fmt(goal_node)

        # Collect all active nodes to declare as PDDL objects
        objects = {init_obj, goal_obj}
        for u, v in self.transitional_map.edges():
            objects.add(_fmt(self._get_merged_node(u)))
            objects.add(_fmt(self._get_merged_node(v)))

        pddl = [f"(define (problem {problem_name})"]
        pddl.append(f"  (:domain {domain_name})")
        pddl.append("  (:objects")
        pddl.append(f"    {' '.join(sorted(objects))}")
        pddl.append("  )")

        # Initial state: agent starting location + transitional graph connectivity
        pddl.append("  (:init")
        pddl.append(f"    (at {init_obj})")

        seen_edges = set()
        for u, v, data in self.transitional_map.edges(data=True):
            u_obj = _fmt(self._get_merged_node(u))
            v_obj = _fmt(self._get_merged_node(v))
            skill = data.get('skill')
            if skill:
                edge_fact = f"can_{skill} {u_obj} {v_obj}"
                if edge_fact not in seen_edges:
                    seen_edges.add(edge_fact)
                    pddl.append(f"    ({edge_fact})")

        pddl.append("  )")

        # Goal condition
        pddl.append("  (:goal (and")
        pddl.append(f"    (at {goal_obj})")
        pddl.append("  ))")
        pddl.append(")")

        return "\n".join(pddl)