import os
import json
import numpy as np

from .predicate import Predicate

class EdgeError(Exception):
    # call closing of evething #TODO#
    pass

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
                    self.count -= 1
                else:
                    raise EdgeError(f"Unsupported operation: {operation}")

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
        self.max_points = 100           # maxium of points in single GNG, if None then 2*dim
        self.base_radius = 0.1          # radius around separated GNG node
        self.max_radius = 1.0           # maximum local radius
        self.learning_rate_b = 0.1      # Fraction to move the nearest node
        self.learning_rate_n = 0.001    # Fraction to move topological neighbors
        self.max_edge_age = 10          # Maximum age of an edge before removal
        self.lambda_step = 1            # Steps between node insertion
        self.alpha = 0.5                # Error reduction during insertion
        self.d = 0.999                  # Global error decay per step

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
            prev_skill_name, prev_vector_state, skill_name, vector_state, succeed = data
            
            # Dynamically initialize state_dim if this is the first data
            if self.state_dim is None:
                self.state_dim = len(prev_vector_state)
            if self.max_points is None:
                self.max_points = self.state_dim * 2

            # Ensure the node exists
            if skill_name not in self.trans_graph.nodes:
                predicate = Predicate(self.max_points, self.base_radius, self.max_radius, 
                                      self.learning_rate_b, self.learning_rate_n, self.max_edge_age, 
                                      self.lambda_step, self.alpha, self.d)
                self._add_node(skill_name, predicate, [])

            # -> update predicate
            self.trans_graph.update_predicate(skill_name, prev_vector_state, succeed)
            
            # -> update edges
            if succeed:
                if not prev_skill_name == skill_name: # do not add self loops
                    self.trans_graph.add_edge(prev_skill_name, skill_name)
            else:
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
        if "edges" in data:
            parsed_edges = {}
            for k, v in data["edges"].items():
                u, v_node = map(int, k.split(","))
                parsed_edges[(u, v_node)] = v
            pred.edges = parsed_edges
        return pred

    def resolve_predicates(self, state_vector: np.ndarray) -> list:
        """
        Resolves a continuous state vector into active PDDL atomic propositions:
        Returns list of active symbols prefixed with 'node_'.
        """
        active_skills = self.get_active_skills(state_vector)
        return [f"node_{skill}" for skill in active_skills]

    def generate_domain_pddl(self, domain_name="world") -> str:
        """
        Generates domain.pddl.
        """
        nodes = self.trans_graph.get_nodes_list()
        
        pddl = [
            f"(define (domain {domain_name})",
            "  (:requirements :typing)",
            "  (:types"
        ]
        
        # We must explicitly declare 'node' before making other types inherit from it!
        if nodes:
            pddl.append("    node")
            types_str = " ".join([f"type_{n}" for n in nodes])
            pddl.append(f"    {types_str} - node")
        else:
            pddl.append("    node")
            
        pddl.extend([
            "  )",
            "  (:predicates",
            "    (at ?n - node)",
            "    (connected ?from - node ?to - node)",
            "  )"
        ])
        
        # Actions restrict the ?to parameter to their strictly associated type
        for node in nodes:
            pddl.extend([
                "",
                f"  (:action {node}",
                f"    :parameters (?from - node ?to - type_{node})",
                "    :precondition (and (at ?from) (connected ?from ?to))",
                "    :effect (and (not (at ?from)) (at ?to))",
                "  )"
            ])
            
        pddl.append(")")
        return "\n".join(pddl)

    def generate_problem_pddl(self, initial_state, goal_state, 
                              problem_name="plan_problem", domain_name="world") -> str:
        """
        Generates problem.pddl.
        """
        nodes = self.trans_graph.get_nodes_list()
        
        pddl = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:objects"
        ]
        
        if nodes:
            pddl.append("    " + " ".join([f"node_{n}" for n in nodes]) + " - node")
        pddl.append("  )")
        pddl.append("  (:init")
        
        # Starting point
        if initial_state in nodes:
            pddl.append(f"    (at node_{initial_state})")
            
        # Draw the graph connections based on active edges
        for target, sources in self.trans_graph.edges_by_target.items():
            for source, edge in sources.items():
                if edge.count > 0:
                    pddl.append(f"    (connected node_{source} node_{target})")
                    
        pddl.append("  )")
        pddl.append("  (:goal")
        pddl.append(f"    (at node_{goal_state})")
        pddl.append("  )")
        pddl.append(")")
        
        return "\n".join(pddl)