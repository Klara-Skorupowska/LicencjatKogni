import os
import json
import numpy as np
import networkx as nx
from collections import defaultdict

class BrainNetwork:
    def __init__(self, log_dir=None):
        self.state_dim = None 

        # Logging Setup
        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.symbols_dir = os.path.join(self.log_dir, "PDDL", "symbols")
            self.graphs_dir = os.path.join(self.log_dir, "graphs")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.symbols_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)

    def classify(self, vector):
        pass

    def predict(self, state, action):
        pass

    def update(self):
        pass

    def _log_graph(self):
        """Saves JSON/GraphML for the graph"""
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
        Builds grounded symbols and persists them to disk.
        """
        pass

    def resolve_predicates(self, state_vector: np.ndarray) -> list:
        """
        Resolves a continuous state vector into active PDDL atomic propositions:
        Returns list of active symbols.
        """
        pass

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