import os
import json
import time
import numpy as np

from .predicate import Predicate

class FatalError(Exception):
    def __init__(self, message):
        super().__init__(message)

class BrainNetwork:
    def __init__(self, log_dir=None):
        self.state_dim = None 
        # helper classes
        class Node:
            def __init__(self, name: str, precondition: Predicate, effect: Predicate, mask = None):
                self.name = name
                self.precondition = precondition
                self.effect = effect
                self.mask = mask if mask is not None else [] # changing dimentions
            
            def __repr__(self):
                return f"Node({self.name})"

        class Edge:
            def __init__(self, source_node: str, target_node: str):
                self.source_node = source_node
                self.target_node = target_node
                self.symbol = None

            def __repr__(self):
                return f"Edge({self.source_node} -> {self.target_node}, count={self.count})"
            
            def update(self, symbol: Predicate) -> bool:
                """
                replaces old symbol with new one
                """
                self.symbol = symbol

        class Graph:
            def __init__(self):
                # Maps node_name -> Node object
                self.nodes = {}
                # Maps target_node -> {source_node: Edge}
                self.edges_by_target = {}

            def add_node(self, name: str, precondition: Predicate, effect: Predicate, mask):
                """Registers a new node in the graph."""
                if name in self.nodes:
                    raise ValueError(f"Node '{name}' already exists.")
                self.nodes[name] = Node(name, precondition, effect, mask)
                if name not in self.edges_by_target:
                    self.edges_by_target[name] = {}

            def add_edge(self, from_name: str, to_name: str):
                """O(1) addition/increment of directed edge: from_name -> to_name"""
                if from_name is None or to_name is None:
                    return
                if from_name not in self.nodes or to_name not in self.nodes:
                    raise KeyError(f"Both nodes must exist: {from_name}, {to_name}")

                incoming = self.edges_by_target.setdefault(to_name, {})
                if not from_name in incoming:
                    incoming[from_name] = Edge(from_name, to_name)

            def delete_edge(self, from_name: str, to_name: str):
                """O(1) decrement/removal of directed edge: from_name -> to_name"""
                incoming = self.edges_by_target.get(to_name)
                if not incoming or from_name not in incoming:
                    return

                del incoming[from_name]

            def get_incoming_edges(self, target_name: str) -> dict:
                """Returns {source_node: Edge} incoming into target_name."""
                return self.edges_by_target.get(target_name, {})

            def get_node(self, name: str):
                return self.nodes.get(name)

            def get_nodes_list(self):
                return list(self.nodes.keys())

            def is_active(self, node_name: str, vector, mask=None) -> bool:
                """Checks if the precondition predicate of a specific node is active in given vector."""
                if node_name not in self.nodes:
                    raise KeyError(f"Node '{node_name}' does not exist.")
                target_mask = mask if mask is not None else self.nodes[node_name].mask
                return self.nodes[node_name].precondition.is_active(vector, mask=target_mask)

            def update_precondition(self, node_name: str, vector, valence: bool):
                """Updates the precondition predicate of a specific node with a new vector and valence."""
                if node_name not in self.nodes:
                    raise KeyError(f"Node '{node_name}' does not exist.")
                self.nodes[node_name].precondition.update(vector, valence)

            def update_effect(self, node_name: str, vector, valence: bool):
                """Updates the effect predicate of a specific node with a new vector and valence."""
                if node_name not in self.nodes:
                    raise KeyError(f"Node '{node_name}' does not exist.")
                if valence: # if action was wrong we do not know nothing about effect
                    self.nodes[node_name].effect.update(vector, True)

        # transitional graph - parameters
        self.trans_graph = Graph()

        # predicates as GNG - parameters
        self.max_points = 100          # maxium of points in single GNG, if None then 2*dim
        self.base_radius = 0.1          # radius around separated GNG node
        self.max_radius = None          # maximum local radius, if None then sqrt(dim)
        self.learning_rate_b = 0.5      # Fraction to move the nearest node
        self.learning_rate_n = 0.1      # Fraction to move topological neighbors
        self.max_edge_age = 10          # Maximum age of an edge before removal
        self.lambda_step = 5            # Steps between node insertion
        self.alpha = 0.5                # Error reduction during insertion
        self.d = 0.99                   # Global error decay per step

        # Logging Setup
        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.gas_dir = os.path.join(self.log_dir, "PDDL", "GNG")
            self.graphs_dir = os.path.join(self.log_dir, "graphs")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.gas_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)

    
    def add_node(self, skill_name: str, precondition: Predicate, effect: Predicate, mask):
        self.trans_graph.add_node(skill_name, precondition, effect, mask)
            
    def create_node(self, skill_name):
        if skill_name not in self.trans_graph.nodes:
            precondition = Predicate(self.max_points, self.base_radius, self.max_radius, 
                                    self.learning_rate_b, self.learning_rate_n, self.max_edge_age, 
                                    self.lambda_step, self.alpha, self.d)
            effect = Predicate(self.max_points, self.base_radius, self.max_radius, 
                                    self.learning_rate_b, self.learning_rate_n, self.max_edge_age, 
                                    self.lambda_step, self.alpha, self.d)
            self.trans_graph.add_node(skill_name, precondition, effect, [])

    def delete_edge(self, from_name: str, to_name: str):
        """
        Removes edge from transition graph and cleans up its pass predicate JSON files.
        """
        # 1. Remove edge from the graph representation
        self.trans_graph.delete_edge(from_name, to_name)

        # 2. Clean up saved predicate file on disk if logging is active
        if self.log_dir:
            symbol_name = f"sym_{from_name}_enables_{to_name}_pass"
            file_path = os.path.join(self.gas_dir, f"{symbol_name}.json")
            temp_file = os.path.join(self.gas_dir, f"{symbol_name}.tmp")

            for path in (file_path, temp_file):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def preconditions_met(self, skill_name, state_vector) -> bool:
        """
        check if predicate is active. GNG of skill_name, check if state_vector is ok
        """
        if self.state_dim is None:
            self.state_dim = len(state_vector)
        if self.max_points is None:
            self.max_points = self.state_dim
        if self.max_radius is None:
            self.max_radius = self.state_dim ** (0.5)

        if skill_name not in self.trans_graph.nodes:
            return False  # New predicate has no vectors 
            
        node = self.trans_graph.nodes[skill_name]
        target_mask = node.mask
        return node.precondition.is_active(state_vector, mask=target_mask)

    def get_active_skills(self, vector):
        """
        get all nodes names, then check which ones are active for given vector
        return a list of active ones
        """
        if self.state_dim is None:
            self.state_dim = len(vector)
        if self.max_points is None:
            self.max_points = self.state_dim
        if self.max_radius is None:
            self.max_radius = self.state_dim ** (0.5)

        active_skills = []
        for name, node in self.trans_graph.nodes.items():
            if node.precondition.is_active(vector):
                active_skills.append(name)
        return active_skills


    def get_nodes(self):
        """
        return all nodes names as a list of strings
        """
        return self.trans_graph.get_nodes_list()

    def update(self, buffor):
        """
        Updates GNG node predicates and transitions using unexpected outcomes accumulated in the buffer.
        """
        print(f"\t[Brain] Updating Predicates (GNG)")
        for data in buffor:
            pre_vector_state, skill_name, post_vector_state, succeed = data
        
            # Ensure state dimensions are initialized
            if self.state_dim is None:
                self.state_dim = len(pre_vector_state)
            if self.max_points is None:
                self.max_points = self.state_dim
            if self.max_radius is None:
                self.max_radius = self.state_dim ** 0.5

            # Ensure target node exists
            if skill_name not in self.trans_graph.nodes:
                self.create_node(skill_name)

            # Update GNG predicates
            self.trans_graph.update_precondition(skill_name, pre_vector_state, succeed)
            self.trans_graph.update_effect(skill_name, post_vector_state, succeed)

            # update transitions
                # from current skill
            self.update_transitions(skill_name)
                # to current skill
            incoming_sources = list(self.trans_graph.get_incoming_edges(skill_name).keys())
            for src_skill in incoming_sources:
                self.update_transitions(src_skill)


        self._logger()
            
    def update_transitions(self, skill_name: str):
        """
        Updates transitional edges outgoing from given skill using its effect and preconditions of other nodes.
        Deletes existing edges that no longer have overlapping points.
        """
        alpha_threshold = 1
        self.compute_gng_mask(skill_name, alpha_threshold)

        src_node = self.trans_graph.nodes[skill_name]
        eff_gng = src_node.effect
        
        # Identify all existing targets that src_node points to
        existing_targets = [
            target_name 
            for target_name, sources in self.trans_graph.edges_by_target.items() 
            if skill_name in sources
        ]

        # If effect has no nodes, delete all outgoing edges and return
        if not eff_gng.nodes:
            for target_skill in existing_targets:
                self.delete_edge(skill_name, target_skill)
            return {}

        overlapping_points_map = {}
        for target_skill, target_node in self.trans_graph.nodes.items():
            init_gng = target_node.precondition
            if not init_gng.nodes:
                # Target precondition is empty: prune edge if it existed
                if target_skill in existing_targets:
                    self.delete_edge(skill_name, target_skill)
                continue

            # Find common masked points (points in eff that fall inside init)
            overlapping_points = []
            for node_idx, node in enumerate(eff_gng.nodes):
                if src_node.mask:
                    init_nodes_sub = np.asarray(init_gng.nodes)[:, src_node.mask]
                    node_sub = node[src_node.mask]
                    dists = np.linalg.norm(init_nodes_sub - node_sub, axis=1)
                    closest_idx = int(np.argmin(dists))
                    radius = init_gng._get_local_radius(closest_idx)
                    is_inside = dists[closest_idx] <= radius
                else:
                    is_inside = init_gng.is_active(node)

                if is_inside:
                    overlapping_points.append(node)

            enough_overlapping = len(overlapping_points) >= len(init_gng.nodes) * 0.6 # filter out too small overlappings
            if overlapping_points and skill_name != target_skill and  enough_overlapping:
                self.trans_graph.add_edge(skill_name, target_skill)
                overlapping_points_map[target_skill] = overlapping_points
            else:
                # No overlap: delete the edge if it previously existed
                if target_skill in existing_targets:
                    self.delete_edge(skill_name, target_skill)

        return overlapping_points_map
    
    def compute_gng_mask(self, skill_name, alpha_threshold=1.0):
        """
        Computes the modified variable mask between an initiation GNG and an effect GNG.
        """
        gng_init = self.trans_graph.nodes[skill_name].precondition
        gng_eff = self.trans_graph.nodes[skill_name].effect

        if not gng_init.nodes or not gng_eff.nodes:
            return [] #Both predicates must contain fitted nodes.

        nodes_init = np.asarray(gng_init.nodes)
        nodes_eff = np.asarray(gng_eff.nodes)
    
        num_dims = nodes_init.shape[1]
        displacements = []
        local_thresholds = []

        # 1. For each node in gng_eff, find its nearest neighbor in gng_init
        for u_idx, u in enumerate(nodes_eff):
            dists = np.linalg.norm(nodes_init - u, axis=1)
            v_idx = int(np.argmin(dists))
            v = nodes_init[v_idx]

            # Component-wise absolute displacement along each dimension
            diff = np.abs(u - v)
            displacements.append(diff)

            # Baseline noise floor: average local radii of the matched nodes
            r_eff = gng_eff._get_local_radius(u_idx)
            r_init = gng_init._get_local_radius(v_idx)
            local_thresholds.append(0.5 * (r_eff + r_init))

        displacements = np.asarray(displacements)           # Shape: (N_eff, num_dims)
        local_thresholds = np.asarray(local_thresholds)     # Shape: (N_eff,)

        # 2. Average absolute displacement across all mapped nodes per dimension
        mean_dim_disp = np.mean(displacements, axis=0)       # Shape: (num_dims,)
    
        # 3. Mean topological boundary scale (Voronoi cell scale)
        mean_radius = np.mean(local_thresholds)
        threshold = alpha_threshold * mean_radius

        # 4. Extract dimensions exceeding the structural neighborhood threshold
        mask = [d for d in range(num_dims) if mean_dim_disp[d] > threshold]

        self.trans_graph.nodes[skill_name].mask = mask

    def abstract_symbols(self):
        '''
        Abstracts symbols from GNGs. 
        '''
        alpha_threshold = 1
        for skill_name in self.trans_graph.nodes.keys():
            self.compute_gng_mask(skill_name, alpha_threshold)

        all_overlaps = {}
        for src_skill in self.trans_graph.nodes.keys():
            all_overlaps[src_skill] = self.update_transitions(src_skill)

        for target_skill, sources in self.trans_graph.edges_by_target.items():
            for src_skill, edge in sources.items():
                init_gng = self.trans_graph.nodes[src_skill].precondition
                overlapping_points = all_overlaps.get(src_skill, {}).get(target_skill, [])
                # Ground a new abstract predicate for the intersection manifold
                sym_pred = Predicate(
                    max_points=max(10, len(overlapping_points)),
                    base_radius=init_gng.base_radius,
                    max_radius=init_gng.max_radius
                )
                for pt in overlapping_points:
                    sym_pred.update(pt, valence=True)

                edge.symbol = sym_pred

    def generate_domain_pddl(self, domain_name="robot_domain") -> str:
        """
        Compiles discovered skills and initiation/effect overlaps into PDDL operators.
        """
        pddl = [
            f"(define (domain {domain_name})",
            "  (:requirements :strips)",
            "  (:predicates"
        ]
        
        # State conditions: Can we start a skill? Did a skill finish?
        for skill in self.trans_graph.nodes.keys():
            pddl.append(f"    (can_run_{skill})")
            pddl.append(f"    (executed_{skill})")
            
        pddl.append("  )")

        # Operators derived from each option
        for src_name in self.trans_graph.nodes.keys():

            targets = [
                target
                for target, sources in self.trans_graph.edges_by_target.items()
                if src_name in sources
            ]

            pddl.extend([
                "",
                f"  (:action {src_name}",
                "    :parameters ()",
                f"    :precondition (can_run_{src_name})",
            ])

            # Building Add & Delete lists based on effects
            adds = [f"(executed_{src_name})"]
            for target in targets:
                adds.append(f"(can_run_{target})")

            # Consumes its own initiation precondition
            deletes = [f"(can_run_{src_name})"]

            pddl.append(f"    :effect (and {' '.join(adds)} (not {' '.join(deletes)}))")
            pddl.append("  )")

        pddl.append(")")
        return "\n".join(pddl)

    def generate_problem_pddl(self, initial_skill, goal_skill, 
                              problem_name="plan_problem", domain_name="robot_domain") -> str:
        """
        Generates problem.pddl.
        """
        pddl = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:init"
        ]
        
        pddl.append(f"    (can_run_{initial_skill})")
            
        pddl.append("  )") 
        pddl.append("  (:goal")
        pddl.append(f"    (executed_{goal_skill})")
        pddl.append("  )")
        pddl.append(")")
        
        return "\n".join(pddl)

    def _logger(self):
        '''
        logs the graph and predicates into a JSON files
        '''
        if not self.log_dir:
            return

        
        # transitions up to date:
        for skill in self.trans_graph.nodes.keys():
            self.update_transitions(skill)
        # symbols up to date:
        self.abstract_symbols()


        for skill, node in self.trans_graph.nodes.items():
            self.save_predicate(skill, node.precondition, 'init')
            self.save_predicate(skill, node.effect, 'eff')

        edges_list = []
        for target, sources in self.trans_graph.edges_by_target.items():
            for source, edge in sources.items():
                symbol_name = ""
                if edge.symbol is not None:
                    symbol_name = f"sym_{source}_enables_{target}"
                    self.save_predicate(symbol_name, edge.symbol, 'pass')
                edges_list.append({
                    "source": edge.source_node,
                    "target": edge.target_node,
                    "symbol_name": symbol_name
                })

        masks = [node.mask for node in self.trans_graph.nodes.values()]
        graph_data = {
            "nodes": list(self.trans_graph.nodes.keys()),
            "masks": masks,
            "edges": edges_list
        }
            
        file_path = os.path.join(self.graphs_dir, "trans_graph.json")
        temp_file = os.path.join(self.graphs_dir, f"trans_graph.tmp")
        with open(temp_file, "w") as f:
            json.dump(graph_data, f)
        for _ in range(5):
            try:
                os.replace(temp_file, file_path)
                break
            except (PermissionError, OSError):
                time.sleep(0.05)
        else:
            # Clean up temp file if all retries fail
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def save_predicate(self, skill_name: str, predicate: Predicate, pred_type: str):
        """
        Saves GNG nodes, edges, local radiuses, and GNG parameters to JSON.
        pred_type - init (init) or effect (eff) or symbol for passing between skills (pass)
        """
        if not self.log_dir:
            return
        if predicate is None:
            return
        if pred_type not in ['init', 'eff', 'pass']:
            return

        os.makedirs(self.gas_dir, exist_ok=True)

        # Collect parameters from self
        parameters = {
            "max_points": self.max_points,
            "base_radius": self.base_radius,
            "max_radius": self.max_radius,
            "learning_rate_b": self.learning_rate_b,
            "learning_rate_n": self.learning_rate_n,
            "max_edge_age": self.max_edge_age,
            "lambda_step": self.lambda_step,
            "alpha": self.alpha,
            "d": self.d,
        }
            
        pred = predicate
        # Edges are stored as tuples like (0, 1) -> Convert to string key for JSON
        edges_str_keys = {f"{u},{v}": age for (u, v), age in pred.edges.items()}
            
        data = {
            "parameters": parameters,
            "nodes": [n.tolist() for n in pred.nodes],
            "edges": edges_str_keys,
            "local_radiuses": [float(pred._get_local_radius(i)) for i in range(len(pred.nodes))]
        }
        file_path = os.path.join(self.gas_dir, f"{skill_name}_{pred_type}.json")
        temp_file = os.path.join(self.gas_dir, f"{skill_name}_{pred_type}.tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f)
        for _ in range(5):
            try:
                os.replace(temp_file, file_path)
                break
            except (PermissionError, OSError):
                time.sleep(0.05)
        else:
            # Clean up temp file if all retries fail
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    @staticmethod
    def load_predicate(file_path) -> Predicate:
        """
        Reads a saved predicate file and reconstructs a Predicate object.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # Read parameters from JSON
        params = data.get("parameters", {})
        pred = Predicate(
            max_points=params["max_points"],
            base_radius=params["base_radius"],
            max_radius=params["max_radius"],
            learning_rate_b=params["learning_rate_b"],
            learning_rate_n=params["learning_rate_n"],
            max_edge_age=params["max_edge_age"],
            lambda_step=params["lambda_step"],
            alpha=params["alpha"],
            d=params["d"]
        )

        if "nodes" in data:
            pred.nodes = [np.array(n) for n in data["nodes"]]
            pred.errors = [0.0] * len(pred.nodes) 
            
        if "edges" in data:
            parsed_edges = {}
            for k, v in data["edges"].items():
                u, v_node = map(int, k.split(","))
                parsed_edges[(u, v_node)] = v
            pred.edges = parsed_edges
            
        return pred

    def load_brain(self, source_dir):
        '''
        loads transgraph an all the predicates from given directory in 'logs'.
        '''
        print(f"\t[Brain] Loading brain from the files.")
        if source_dir is None:
            print(f"\t[Brain] No source directory.")
            return
        
        if os.path.isdir(source_dir):
            # 1. Define source paths
            src_graphs_file = os.path.join(source_dir, "graphs", "trans_graph.json")
            src_gas_dir = os.path.join(source_dir, "PDDL", "GNG")
            
            # 2. Define target paths (directories are already created in __init__)
            target_graphs_file = os.path.join(self.graphs_dir, "trans_graph.json")
            target_gas_dir = self.gas_dir
            
            # 3. Copy trans_graph.json
            if not os.path.exists(src_graphs_file):
                print(f"\t[Brain] Could not find trans_graph.json in {source_dir}")
                return
                
            with open(src_graphs_file, "r") as src_f, open(target_graphs_file, "w") as dst_f:
                dst_f.write(src_f.read())
                
            # 4. Copy all symbol JSON files
            if os.path.exists(src_gas_dir):
                for filename in os.listdir(src_gas_dir):
                    if filename.endswith(".json"):
                        src_file = os.path.join(src_gas_dir, filename)
                        dst_file = os.path.join(target_gas_dir, filename)
                        with open(src_file, "r") as src_f, open(dst_file, "w") as dst_f:
                            dst_f.write(src_f.read())

            # 5. Load the graph data from the NEW target location
            with open(target_graphs_file, "r") as f:
                graph_data = json.load(f)
                
            # 6. Load nodes and their predicates from the NEW target location
            masks = graph_data.get("masks", [])
            for idx, node_name in enumerate(graph_data.get("nodes", [])):
                
                precondition_file = os.path.join(target_gas_dir, f"{node_name}_init.json")
                if os.path.exists(precondition_file):
                    precondition= self.load_predicate(precondition_file)
                else:
                    print(f"\t[Brain] Warning: Missing precondition file for node {node_name}")
                
                effect_file = os.path.join(target_gas_dir, f"{node_name}_eff.json")
                if os.path.exists(effect_file):
                    effect= self.load_predicate(effect_file)
                else:
                    print(f"\t[Brain] Warning: Missing effect file for node {node_name}")

                if precondition and effect:
                    self.add_node(node_name, precondition, effect, masks[idx])    
            
            
            # 7. Reconstruct edges and their symbols
            for edge in graph_data.get("edges", []):
                source = edge.get("source")
                target = edge.get("target")
                symbol_name = edge.get("symbol_name")

                self.trans_graph.add_edge(source, target)

                symbol_file = os.path.join(target_gas_dir, f"{symbol_name}_pass.json")
                if os.path.exists(symbol_file):
                    symbol= self.load_predicate(symbol_file)
                    self.trans_graph.edges_by_target[target][source].symbol = symbol
                else:
                    print(f"\t[Brain] Warning: Missing symbol file for edge: {source} --> {target}")

                    