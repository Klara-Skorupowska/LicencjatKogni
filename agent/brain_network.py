import os
import json
import numpy as np

from .predicate import Predicate

class BrainNetwork:
    def __init__(self, log_dir=None):
        self.state_dim = None 
        # helper classes
        class Node:
            def __init__(self, name: str, predicate: Predicate):
                self.name = name
                self.predicate = predicate
                # Adjacency list: holds the names of nodes this node vectors to
                self.targets = [] 

            def add_target(self, target_name: str):
                """Adds a directed edge from this node to the target node."""
                if target_name not in self.targets:
                    self.targets.append(target_name)

            def remove_target(self, target_name: str):
                """Removes a directed edge from this node."""
                if target_name in self.targets:
                    self.targets.remove(target_name)
            
            def __repr__(self):
                return f"Node({self.name}, targets={self.targets})"

        class Graph:
            def __init__(self):
                # Maps node_name -> Node object
                self.nodes = {}

            def add_node(self, name: str, predicate):
                """Registers a new node in the graph."""
                if name in self.nodes:
                    raise ValueError(f"Node '{name}' already exists.")
                self.nodes[name] = Node(name, predicate)

            def add_edge(self, from_name: str, to_name: str):
                """Creates a directed edge: from_name -> to_name"""
                if from_name not in self.nodes or to_name not in self.nodes:
                    raise KeyError("Both nodes must exist in the graph before adding an edge.")
        
                self.nodes[from_name].add_target(to_name)

            def get_node(self, name: str):
                return self.nodes.get(name)

            def get_nodes_list(self):
                return list(self.nodes.keys())

            def delete_edge(self, from_name: str, to_name: str):
                """Removes a directed edge: from_name -> to_name"""
                if from_name not in self.nodes:
                    raise KeyError(f"Node '{from_name}' does not exist.")
                # We don't necessarily need to check if to_name exists in self.nodes, 
                # just if it's in the target list of from_name, which remove_target handles.
                self.nodes[from_name].remove_target(to_name)

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

        # predicates as GNG
        self.max_points = None              # maximum of vectors in one GNG if None it calculates it as 2*dimention
        self.tolerance = 0.01               # maximum distance from the vector
        self.max_edge_age = 50              # for deleting unvisited regions
        self.learning_rate = 0.2    

        # Logging Setup
        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.symbols_dir = os.path.join(self.log_dir, "PDDL", "symbols")
            self.graphs_dir = os.path.join(self.log_dir, "graphs")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.symbols_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)

    def _add_node(self, name: str, predicate: Predicate, incoming_actions: list[str]):
        '''
        adds node to the graph, name as a dict key, then in node: name, predicate. Then adds incoming_actions.
        '''
        if name not in self.trans_graph.nodes:
            self.trans_graph.add_node(name, predicate)
        
        for action in incoming_actions:
            if action in self.trans_graph.nodes:
                self.trans_graph.add_edge(action, name)

    def _update_incoming_actions(self, skill_name, vector_state, success):
        """
        updates incoming actions. if failed then delete prev_skill from skill's node. if success add it. do not duplicate the edges
        """
        if vector_state is None or skill_name is None:
            return
        # 
        active_skills = self.get_active_skills(vector_state)
        # Ensure each node exist
        for possible_skill in active_skills:
            if possible_skill not in self.trans_graph.nodes:
                self._add_node(possible_skill, Predicate(self.max_points), [])
        if skill_name not in self.trans_graph.nodes:
            self._add_node(skill_name, Predicate(self.max_points), [])

        if success:
            for possible_skill in active_skills:
                self.trans_graph.add_edge(possible_skill, skill_name)
        else:
            try:
                for possible_skill in active_skills:
                    self.trans_graph.delete_edge(possible_skill, skill_name)
            except KeyError:
                pass # Edge didn't exist, which is fine

    def preconditions_met(self, skill_name, state_vector) -> bool:
        """
        check if predicate is active. GNG of skill_name, check if state_vector is ok
        """
        if self.state_dim is None:
            self.state_dim = len(state_vector)
            if self.max_points is None:
                self.max_points = self.state_dim * 2

        if skill_name not in self.trans_graph.nodes:
            self._add_node(skill_name, Predicate(self.max_points), [])
            return False # Brand new predicate has no vectors yet
            
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
            prev_vector_state, skill_name, vector_state, succeed = data
            
            # Dynamically initialize state_dim if this is the first data
            if self.state_dim is None:
                self.state_dim = len(prev_vector_state)
            if self.max_points is None:
                self.max_points = self.state_dim * 2

            # Ensure the node exists
            if skill_name not in self.trans_graph.nodes:
                self._add_node(skill_name, Predicate(self.max_points), [])

            # -> update predicate
            self.trans_graph.update_predicate(skill_name, prev_vector_state, succeed)
            
            # -> update edges
            self._update_incoming_actions(skill_name, vector_state, succeed)

        self._logger()
            

    def _logger(self):
        '''
        logs the graph and predicates into a JSON files
        '''
        if not self.log_dir:
            return

        self.create_predicates()

        graph_data = {}
        for name, node in self.trans_graph.nodes.items():
            graph_data[name] = node.targets
            
        file_path = os.path.join(self.graphs_dir, "trans_graph.json")
        with open(file_path, "w") as f:
            json.dump(graph_data, f, indent=4)

    def create_predicates(self):
        """
        Builds grounded symbols and persists them to disk. 
        Saves GNG nodes, edges, and local radiuses to JSON.
        """
        if not self.log_dir:
            return
            
        for name, node in self.trans_graph.nodes.items():
            pred = node.predicate
            # Edges are stored as tuples like (0, 1) -> Convert to string key for JSON
            edges_str_keys = {f"{u},{v}": age for (u, v), age in pred.edges.items()}
            
            data = {
                "nodes": [n.tolist() for n in pred.nodes],
                "edges": edges_str_keys,
                "local_radiuses": [pred._get_local_radius(i) for i in range(len(pred.nodes))]
            }
            file_path = os.path.join(self.symbols_dir, f"{name}.json")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)

    @staticmethod
    def load_predicate(file_path, max_vectors) -> Predicate:
        """
        Reads a saved predicate file and reconstructs a Predicate object.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
            
        pred = Predicate(max_vectors)
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
        Returns list of active symbols.
        """
        return []

    def generate_domain_pddl(self, domain_name="world") -> str:
        """
        Generates domain.pddl.
        """
        return ""

    def generate_problem_pddl(self, initial_state, goal_state, 
                              problem_name="plan_problem", domain_name="world") -> str:
        """
        Generates problem.pddl.
        """
        return ""