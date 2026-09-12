import os
import json
import time
import numpy as np

from .predicate import Predicate

# helper classes

class FatalError(Exception):
    def __init__(self, message):
        super().__init__(message)

class Action:
    def __init__(self, name: str):
        self.name = name
        self.mask = []  # dimentions handled by this action
        self.init_state = Predicate(
            max_points = 100,           # maxium of points in single GNG, if None then 2*dim
            base_radius = 0.1,          # radius around separated GNG node
            max_radius = 15,            # maximum local radius, if None then sqrt(dim)
            learning_rate_b = 0.2,      # Fraction to move the nearest node
            learning_rate_n = 0.01,     # Fraction to move topological neighbors
            max_edge_age = 10,          # Maximum age of an edge before removal
            lambda_step = 3,            # Steps between node insertion
            alpha = 0.5,                # Error reduction during insertion
            d = 0.95,                   # Global error decay per step
            predicate_type= 'init'
        )
        self.effect_state = Predicate(
            max_points = 100,           # maxium of points in single GNG, if None then 2*dim
            base_radius = 0.1,          # radius around separated GNG node
            max_radius = 15,            # maximum local radius, if None then sqrt(dim)
            learning_rate_b = 0.2,      # Fraction to move the nearest node
            learning_rate_n = 0.01,     # Fraction to move topological neighbors
            max_edge_age = 10,          # Maximum age of an edge before removal
            lambda_step = 3,            # Steps between node insertion
            alpha = 0.5,                # Error reduction during insertion
            d = 0.95,                   # Global error decay per step
            predicate_type= 'effect'
        )
    def preconditions_met(self, vector) -> bool:
        return self.init_state.is_active(vector)

class BrainNetwork:
    def __init__(self, log_dir=None):
        self.state_dim = None 
        self.actions = {}
        self.symbols = {}          # Discovered grounded symbols: name -> Predicate
        self.overlap_map = {}      # eff_name -> [init_names enabled]

        # Logging Setup
        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.symbols_dir = os.path.join(self.log_dir, "PDDL", "symbols")
            self.gngs_dir = os.path.join(self.log_dir, "GNG")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.symbols_dir, exist_ok=True)
            os.makedirs(self.gngs_dir, exist_ok=True)
            
    def add_action(self, name: str) -> Action:
        """
        Registers a new named action with its initiation and effect GNG predicates.
        """
        if name not in self.actions:
            self.actions[name] = Action(name)
        return self.actions[name]

    def get_active_skills(self, vector):
        active_skills = []
        for action in self.actions.values():
            if action.preconditions_met(vector):
                active_skills.append(action)
        return active_skills

    def update_predicates(self, buffor):
        """
        Updates GNG node predicates using transitions accumulated in the buffer:
        (prev_vector_state, skill_name, vector_state, succeed)
        """
        print(f"\t[Brain] Updating Predicates (GNG)")
        for data in buffor:
            prev_vector_state, skill_name, vector_state, succeed = data
            if skill_name not in self.actions:
                self.actions[skill_name] = Action(skill_name)

            self.actions[skill_name].init_state.update(prev_vector_state, succeed)
            self.actions[skill_name].effect_state.update(vector_state, succeed)

        self._logger()

    def _logger(self):
        """
        Logs the raw GNG models into JSON files.
        """
        if not self.log_dir:
            return

        for name, action in self.actions.items():
            for pred in [action.init_state, action.effect_state]:
                self._save_predicate_to_disk(
                    pred, 
                    os.path.join(self.gngs_dir, f"{name}_{pred.pred_type}.json")
                )

    def _save_predicate_to_disk(self, pred: Predicate, file_path: str):
        parameters = {
            "max_points": pred.max_points,
            "base_radius": pred.base_radius,
            "max_radius": pred.max_radius,
            "learning_rate_b": pred.eb,
            "learning_rate_n": pred.en,
            "max_edge_age": pred.max_age,
            "lambda_step": pred.lambda_step,
            "alpha": pred.alpha,
            "d": pred.d,
            "type": pred.pred_type,
        }
        edges_str_keys = {f"{u},{v}": age for (u, v), age in pred.edges.items()}
        data = {
            "parameters": parameters,
            "nodes": [n.tolist() for n in pred.nodes],
            "edges": edges_str_keys,
            "local_radiuses": [float(pred._get_local_radius(i)) for i in range(len(pred.nodes))]
        }
        temp_file = file_path + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(data, f)
        for _ in range(5):
            try:
                os.replace(temp_file, file_path)
                break
            except (PermissionError, OSError):
                time.sleep(0.05)
        else:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    @staticmethod
    def _compute_gng_mask(gng_init, gng_eff, alpha_threshold=1.0):
        """
        Calculates which dimensions change between initiation and termination.
        """
        if not gng_init.nodes or not gng_eff.nodes:
            return [], np.array([])

        nodes_init = np.asarray(gng_init.nodes)
        nodes_eff = np.asarray(gng_eff.nodes)
    
        num_dims = nodes_init.shape[1]
        displacements = []
        local_thresholds = []

        for u_idx, u in enumerate(nodes_eff):
            dists = np.linalg.norm(nodes_init - u, axis=1)
            v_idx = int(np.argmin(dists))
            v = nodes_init[v_idx]

            diff = np.abs(u - v)
            displacements.append(diff)

            r_eff = gng_eff._get_local_radius(u_idx)
            r_init = gng_init._get_local_radius(v_idx)
            local_thresholds.append(0.5 * (r_eff + r_init))

        displacements = np.asarray(displacements)
        local_thresholds = np.asarray(local_thresholds)

        mean_dim_disp = np.mean(displacements, axis=0)
        mean_radius = np.mean(local_thresholds)
        threshold = alpha_threshold * mean_radius

        mask = [d for d in range(num_dims) if mean_dim_disp[d] > threshold]
        return mask, mean_dim_disp

    def create_predicates(self):
        """
        Konidaris Step 3 & 4:
        1. Identifies masks for each option.
        2. Computes pairwise subspace intersections (Eff_i ∩ Init_j).
        3. Induces abstract propositional symbols and builds overlap map.
        """
        print(f"\t[Brain] Discovering Abstract Symbols and Planning Graph")
        self.overlap_map.clear()
        self.symbols.clear()

        # Step 3: Discover individual option masks
        for name, action in self.actions.items():
            mask, _ = self._compute_gng_mask(action.init_state, action.effect_state)
            action.mask = mask

        # Step 4: Inter-option pairwise evaluations
        for src_name, src_action in self.actions.items():
            eff_key = f"eff_{src_name}"
            self.overlap_map[eff_key] = []
            eff_gng = src_action.effect_state
            mask = src_action.mask

            if not eff_gng.nodes:
                continue

            for tgt_name, tgt_action in self.actions.items():
                init_key = f"init_{tgt_name}"
                init_gng = tgt_action.init_state

                if not init_gng.nodes:
                    continue

                overlapping_points = []
                init_nodes_arr = np.asarray(init_gng.nodes)

                for node in eff_gng.nodes:
                    if len(mask) > 0:
                        # Project onto the mask factor subspace
                        init_sub = init_nodes_arr[:, mask]
                        node_sub = node[mask]
                        dists = np.linalg.norm(init_sub - node_sub, axis=1)
                        closest_idx = int(np.argmin(dists))
                        r = init_gng._get_local_radius(closest_idx)
                        if dists[closest_idx] <= r:
                            overlapping_points.append(node)
                    else:
                        if init_gng.is_active(node):
                            overlapping_points.append(node)

                # If overlap exists, an abstract transition is feasible
                if overlapping_points:
                    self.overlap_map[eff_key].append(init_key)

                    symbol_name = f"sym_{src_name}_to_{tgt_name}"
                    sym_pred = Predicate(
                        max_points=max(10, len(overlapping_points)),
                        base_radius=init_gng.base_radius,
                        max_radius=init_gng.max_radius,
                        predicate_type='abstract'
                    )
                    for pt in overlapping_points:
                        sym_pred.update(pt, valence=True)

                    self.symbols[symbol_name] = sym_pred

                    if self.log_dir:
                        file_path = os.path.join(self.symbols_dir, f"{symbol_name}.json")
                        self._save_predicate_to_disk(sym_pred, file_path)

    @staticmethod
    def load_predicate(file_path) -> Predicate:
        """
        Reads a saved predicate file and reconstructs a Predicate object.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
            
        params = data.get("parameters", {})
        pred = Predicate(
            max_points=params.get("max_points", 100),
            base_radius=params.get("base_radius", 0.1),
            max_radius=params.get("max_radius", 15),
            learning_rate_b=params.get("learning_rate_b", 0.2),
            learning_rate_n=params.get("learning_rate_n", 0.01),
            max_edge_age=params.get("max_edge_age", 10),
            lambda_step=params.get("lambda_step", 3),
            alpha=params.get("alpha", 0.5),
            d=params.get("d", 0.95),
            predicate_type=params.get("type", None)
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

            # Rebuild adjacency
            pred.adj = {}
            for u, v_node in pred.edges.keys():
                pred.adj.setdefault(u, set()).add(v_node)
                pred.adj.setdefault(v_node, set()).add(u)
            
        return pred

    def load_brain(self, source_dir):
        """
        Loads GNGs and symbols from an existing log directory into memory.
        """
        print(f"\t[Brain] Loading brain from: {source_dir}")
        if not source_dir or not os.path.isdir(source_dir):
            print(f"\t[Brain] Directory not found.")
            return
        
        src_gngs_dir = os.path.join(source_dir, "GNG")
        src_symbols_dir = os.path.join(source_dir, "PDDL", "symbols")

        # Load option GNGs
        if os.path.exists(src_gngs_dir):
            for filename in os.listdir(src_gngs_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(src_gngs_dir, filename)
                    base_name = filename[:-5]
                    parts = base_name.rsplit("_", 1)
                    if len(parts) == 2:
                        action_name, p_type = parts
                        if action_name not in self.actions:
                            self.actions[action_name] = Action(action_name)
                        
                        pred = self.load_predicate(file_path)
                        if p_type == 'init':
                            self.actions[action_name].init_state = pred
                        elif p_type == 'effect':
                            self.actions[action_name].effect_state = pred

        # Load abstract symbols
        if os.path.exists(src_symbols_dir):
            for filename in os.listdir(src_symbols_dir):
                if filename.endswith(".json"):
                    sym_name = filename[:-5]
                    file_path = os.path.join(src_symbols_dir, filename)
                    self.symbols[sym_name] = self.load_predicate(file_path)

    def generate_domain_pddl(self, domain_name="robot_domain") -> str:
        """
        Konidaris Step 5:
        Compiles discovered skills and initiation/effect overlaps into PDDL operators.
        """
        pddl = [
            f"(define (domain {domain_name})",
            "  (:requirements :strips)",
            "  (:predicates"
        ]
        
        # State conditions: Can we start an action? Did an action finish?
        for act in self.actions.keys():
            pddl.append(f"    (can_run_{act})")
            pddl.append(f"    (executed_{act})")
            
        pddl.append("  )")

        # Operators derived from each option
        for src_name in self.actions.keys():
            eff_key = f"eff_{src_name}"
            enabled_inits = self.overlap_map.get(eff_key, [])

            pddl.extend([
                "",
                f"  (:action {src_name}",
                "    :parameters ()",
                f"    :precondition (can_run_{src_name})",
            ])

            # Building Add & Delete lists based on effects
            adds = [f"(executed_{src_name})"]
            for init_str in enabled_inits:
                target_act = init_str.replace("init_", "")
                adds.append(f"(can_run_{target_act})")

            # Consumes its own initiation precondition
            deletes = [f"(can_run_{src_name})"]

            pddl.append(f"    :effect (and {' '.join(adds)} (not {' '.join(deletes)}))")
            pddl.append("  )")

        pddl.append(")")
        return "\n".join(pddl)

    def generate_problem_pddl(self, current_vector, target_skill, 
                              problem_name="plan_task", domain_name="robot_domain") -> str:
        """
        Grounds the initial state by testing which skills can run on the continuous vector,
        and sets the goal condition to completing the target skill.
        """
        pddl = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:init"
        ]
        
        for name, action in self.actions.items():
            if action.preconditions_met(current_vector):
                pddl.append(f"    (can_run_{name})")
            
        pddl.append("  )") 
        pddl.append("  (:goal")
        pddl.append(f"    (executed_{target_skill})")
        pddl.append("  )")
        pddl.append(")")
        
        return "\n".join(pddl)