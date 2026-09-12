import os
import json
from tkinter import SEL
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
            def __init__(self, name: str, predicate: Predicate):
                self.name = name
                self.predicate = predicate
            
            def __repr__(self):
                return f"Node({self.name})"

        class Edge:
            def __init__(self, source_node: str, target_node: str):
                self.source_node = source_node
                self.target_node = target_node
                self.count = 1

            def __repr__(self):
                return f"Edge({self.source_node} -> {self.target_node}, count={self.count})"
            
            def update(self, operation: str) -> bool:
                """
                operation = 'add' or 'delete'
                """
                if operation == 'add':
                    self.count += 1
                elif operation == 'delete':
                    self.count  = 0 # really delete += - 10 # much harder on the wrong points # or += -1 
                else:
                    raise FatalError(f"Unsupported operation: {operation}")

                return self.count > 0

        class Graph:
            def __init__(self):
                # Maps node_name -> Node object
                self.nodes = {}
                # Maps target_node -> {source_node: Edge}
                self.edges_by_target = {}

            def add_node(self, name: str, predicate):
                """Registers a new node in the graph."""
                if name in self.nodes:
                    raise ValueError(f"Node '{name}' already exists.")
                self.nodes[name] = Node(name, predicate)
                if name not in self.edges_by_target:
                    self.edges_by_target[name] = {}

            def add_edge(self, from_name: str, to_name: str):
                """O(1) addition/increment of directed edge: from_name -> to_name"""
                if from_name is None or to_name is None:
                    return
                if from_name not in self.nodes or to_name not in self.nodes:
                    raise KeyError(f"Both nodes must exist: {from_name}, {to_name}")

                incoming = self.edges_by_target.setdefault(to_name, {})
                if from_name in incoming:
                    incoming[from_name].update('add')
                else:
                    incoming[from_name] = Edge(from_name, to_name)

            def delete_edge(self, from_name: str, to_name: str):
                """O(1) decrement/removal of directed edge: from_name -> to_name"""
                incoming = self.edges_by_target.get(to_name)
                if not incoming or from_name not in incoming:
                    return

                edge = incoming[from_name]
                still_alive = edge.update('delete')
                if not still_alive:
                    del incoming[from_name]

            def get_incoming_edges(self, target_name: str) -> dict:
                """Returns {source_node: Edge} incoming into target_name."""
                return self.edges_by_target.get(target_name, {})

            def get_node(self, name: str):
                return self.nodes.get(name)

            def get_nodes_list(self):
                return list(self.nodes.keys())

            def is_active(self, node_name: str, vector) -> bool:
                """Checks if the predicate of a specific node is active in given vector."""
                if node_name not in self.nodes:
                    raise KeyError(f"Node '{node_name}' does not exist.")
                return self.nodes[node_name].predicate.is_active(vector)

            def update_predicate(self, node_name: str, vector, valence: bool):
                """Updates the predicate of a specific node with a new vector and valence."""
                if node_name not in self.nodes:
                    raise KeyError(f"Node '{node_name}' does not exist.")
                self.nodes[node_name].predicate.update(vector, valence)

        # transitional graph
        self.trans_graph = Graph()

        # predicates as GNG - parameters
        self.max_points = None          # maxium of points in single GNG, if None then 2*dim
        self.base_radius = 0.1          # radius around separated GNG node
        self.max_radius = None          # maximum local radius, if None then sqrt(dim)
        self.learning_rate_b = 0.2      # Fraction to move the nearest node
        self.learning_rate_n = 0.01     # Fraction to move topological neighbors
        self.max_edge_age = 10          # Maximum age of an edge before removal
        self.lambda_step = 5            # Steps between node insertion
        self.alpha = 0.5                # Error reduction during insertion
        self.d = 0.95                   # Global error decay per step

        # Logging Setup
        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.symbols_dir = os.path.join(self.log_dir, "PDDL", "symbols")
            self.graphs_dir = os.path.join(self.log_dir, "graphs")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.symbols_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)

    def create_node(self, skill_name):
        if skill_name not in self.trans_graph.nodes:
            predicate = Predicate(self.max_points, self.base_radius, self.max_radius, 
                                    self.learning_rate_b, self.learning_rate_n, self.max_edge_age, 
                                    self.lambda_step, self.alpha, self.d)
            self._add_node(skill_name, predicate, [])
    
    def _add_node(self, name: str, predicate: Predicate, incoming_actions: list[str]):
        '''
        adds node to the graph, name as a dict key, then in node: name, predicate. Then adds incoming_actions.
        '''
        if name not in self.trans_graph.nodes:
            self.trans_graph.add_node(name, predicate)
        
        for action in incoming_actions:
            if action in self.trans_graph.nodes:
                self.trans_graph.add_edge(action, name)

    def preconditions_met(self, skill_name, state_vector) -> bool:
        """
        check if predicate is active. GNG of skill_name, check if state_vector is ok
        """
        if self.state_dim is None:
            self.state_dim = len(state_vector)
            if self.max_points is None:
                self.max_points = self.state_dim * 2
            if self.max_radius is None:
                self.max_radius = self.state_dim ** (0.5)

        if skill_name not in self.trans_graph.nodes:
            return False # New predicate has no vectors 
            
        return self.trans_graph.nodes[skill_name].predicate.is_active(state_vector)

    def get_active_skills(self, vector):
        """
        get all nodes names, then check which ones are active for given vector
        return a list of active ones
        """
        if self.state_dim is None:
            self.state_dim = len(vector)
            if self.max_points is None:
                self.max_points = self.state_dim * 2
            if self.max_radius is None:
                self.max_radius = self.state_dim ** (0.5)

        active_skills = []
        for name, node in self.trans_graph.nodes.items():
            if node.predicate.is_active(vector):
                active_skills.append(name)
        return active_skills


    def get_nodes(self):
        """
        return all nodes names as a list of strings
        """
        return self.trans_graph.get_nodes_list()

    def update(self, buffor):
        '''
        updates the transitional graph.
        '''
        print(f"\t[Brain] Update Brain")
        for data in buffor:
            prev_skill_name, prev_vector_state, prev_succeed, skill_name, vector_state, succeed = data
            
            # Dynamically initialize state_dim if this is the first data
            if self.state_dim is None:
                self.state_dim = len(prev_vector_state)
                if self.max_points is None:
                    self.max_points = self.state_dim * 2
                if self.max_radius is None:
                    self.max_radius = self.state_dim ** (0.5)

            # Ensure the node exists
            if skill_name not in self.trans_graph.nodes:
                predicate = Predicate(self.max_points, self.base_radius, self.max_radius, 
                                      self.learning_rate_b, self.learning_rate_n, self.max_edge_age, 
                                      self.lambda_step, self.alpha, self.d)
                self._add_node(skill_name, predicate, [])

            # -> update predicate
            self.trans_graph.update_predicate(skill_name, prev_vector_state, succeed)
            
            # -> update edges if previously we were within the node
            if prev_succeed: # prev_succeed == this is not accidental
                if succeed: # it put as in a good place
                    if not prev_skill_name == skill_name: # do not add self loops
                        self.trans_graph.add_edge(prev_skill_name, skill_name)
                else: # it put as in a bad place
                    self.trans_graph.delete_edge(prev_skill_name, skill_name)
            

        self._logger()
            

    def _logger(self):
        '''
        logs the graph and predicates into a JSON files
        '''
        if not self.log_dir:
            return

        self.create_predicates()

        edges_list = []
        for target, sources in self.trans_graph.edges_by_target.items():
            for source, edge in sources.items():
                edges_list.append({
                    "source": edge.source_node,
                    "target": edge.target_node,
                    "count": edge.count
                })

        graph_data = {
            "nodes": list(self.trans_graph.nodes.keys()),
            "edges": edges_list
        }
            
        file_path = os.path.join(self.graphs_dir, "trans_graph.json")
        with open(file_path, "w") as f:
            json.dump(graph_data, f, indent=4)

    def create_predicates(self):
        """
        Builds grounded symbols and persists them to disk.
        Saves GNG nodes, edges, local radiuses, and GNG parameters to JSON.
        """
        if not self.log_dir:
            return

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
            
        for name, node in self.trans_graph.nodes.items():
            pred = node.predicate
            # Edges are stored as tuples like (0, 1) -> Convert to string key for JSON
            edges_str_keys = {f"{u},{v}": age for (u, v), age in pred.edges.items()}
            
            data = {
                "parameters": parameters,
                "nodes": [n.tolist() for n in pred.nodes],
                "edges": edges_str_keys,
                "local_radiuses": [pred._get_local_radius(i) for i in range(len(pred.nodes))]
            }
            file_path = os.path.join(self.symbols_dir, f"{name}.json")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)

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
            src_symbols_dir = os.path.join(source_dir, "PDDL", "symbols")
            
            # 2. Define target paths (directories are already created in __init__)
            target_graphs_file = os.path.join(self.graphs_dir, "trans_graph.json")
            target_symbols_dir = self.symbols_dir
            
            # 3. Copy trans_graph.json
            if not os.path.exists(src_graphs_file):
                print(f"\t[Brain] Could not find trans_graph.json in {source_dir}")
                return
                
            with open(src_graphs_file, "r") as src_f, open(target_graphs_file, "w") as dst_f:
                dst_f.write(src_f.read())
                
            # 4. Copy all symbol JSON files
            if os.path.exists(src_symbols_dir):
                for filename in os.listdir(src_symbols_dir):
                    if filename.endswith(".json"):
                        src_file = os.path.join(src_symbols_dir, filename)
                        dst_file = os.path.join(target_symbols_dir, filename)
                        with open(src_file, "r") as src_f, open(dst_file, "w") as dst_f:
                            dst_f.write(src_f.read())

            # 5. Load the graph data from the NEW target location
            with open(target_graphs_file, "r") as f:
                graph_data = json.load(f)
                
            # 6. Load nodes and their predicates from the NEW target location
            for node_name in graph_data.get("nodes", []):
                predicate_file = os.path.join(target_symbols_dir, f"{node_name}.json")
                if os.path.exists(predicate_file):
                    predicate = self.load_predicate(predicate_file)
                    self._add_node(node_name, predicate, [])
                else:
                    print(f"\t[Brain] Warning: Missing predicate file for node {node_name}")
            
            # 7. Reconstruct edges and their counts
            for edge in graph_data.get("edges", []):
                source = edge.get("source")
                target = edge.get("target")
                count = edge.get("count", 1)
                
                for _ in range(count):
                    self.trans_graph.add_edge(source, target)
    
    def resolve_predicates(self, state_vector: np.ndarray) -> list:
        """
        Resolves a continuous state vector into active PDDL atomic propositions:
        Returns list of active symbols prefixed with 'node_'.
        """
        active_skills = self.get_active_skills(state_vector)
        return [f"node_{skill}" for skill in active_skills]

    def generate_domain_pddl(self, domain_name="world") -> str:
        """
        Generates domain.pddl using propositional logic for multi-target activation.
        Fixed for strict parsers requiring :parameters ().
        """
        nodes = self.trans_graph.get_nodes_list()
        
        pddl = [
            f"(define (domain {domain_name})",
            "  (:predicates"
        ]
        
        # Generate a unique active predicate for every node
        for n in nodes:
            pddl.append(f"    (active_{n})")
            
        pddl.append("  )")
        
        # Generate actions with hardcoded outgoing connections
        for node in nodes:
            # 1. Find all nodes this specific node points to
            outgoing_nodes = []
            for target, sources in self.trans_graph.edges_by_target.items():
                if node in sources and sources[node].count > 0:
                    outgoing_nodes.append(target)
            
            pddl.extend([
                "",
                f"  (:action {node}",
                "    :parameters ()",  # <-- Added this line to satisfy Pyperplan
                f"    :precondition (active_{node})"
            ])
            
            # 2. Build the effects: deactivate current (optional), activate all targets
            effects = [f"(not (active_{node}))"] # Consumes the current active state
            for out in outgoing_nodes:
                effects.append(f"(active_{out})")
                
            if len(effects) == 1:
                pddl.append(f"    :effect {effects[0]}")
            else:
                effects_str = " ".join(effects)
                pddl.append(f"    :effect (and {effects_str})")
                
            pddl.append("  )")
            
        pddl.append(")")
        return "\n".join(pddl)

    def generate_problem_pddl(self, initial_state, goal_state, 
                              problem_name="plan_problem", domain_name="world") -> str:
        """
        Generates problem.pddl.
        """
        pddl = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:init"
        ]
        
        # We only need to declare the initial active state
        if initial_state:
            pddl.append(f"    (active_{initial_state})")
            
        pddl.append("  )") 
        pddl.append("  (:goal")
        pddl.append(f"    (active_{goal_state})")
        pddl.append("  )")
        pddl.append(")")
        
        return "\n".join(pddl)