import numpy as np

# Predicate as GNG (Aligned with Fritzke's Standard GNG Literature)
class Predicate():
    def __init__(self, max_points, base_radius=0.1, max_radius=1.0, 
                 learning_rate_b=0.2, learning_rate_n=0.006, 
                 max_edge_age=5, lambda_step=1, alpha=0.5, d=0.995):
        
        self.max_points = max_points
        self.base_radius = base_radius    
        self.max_radius = max_radius      
        
        # Standard GNG Parameters
        self.eb = learning_rate_b         # Fraction to move the nearest node
        self.en = learning_rate_n         # Fraction to move topological neighbors
        self.max_age = max_edge_age       # Maximum age of an edge before removal
        self.lambda_step = lambda_step    # Steps between node insertion
        self.alpha = alpha                # Error reduction during insertion
        self.d = d                        # Global error decay per step
        
        self.nodes = []                   # Holds the valid vectors
        self.errors = []                  # Holds accumulated error for each node
        self.edges = {}                   # dict of tuple(node_i, node_j) -> age
        self.adj = {}                     # dict of int -> set of neighbor node indices

        self.update_count = 0

    def _add_edge(self, u, v, age=0):
        """Adds or resets an edge and updates the adjacency map."""
        self.edges[tuple(sorted((u, v)))] = age
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set()).add(u)

    def _remove_edge(self, u, v):
        """Removes an edge and updates the adjacency map."""
        edge = tuple(sorted((u, v)))
        if edge in self.edges:
            del self.edges[edge]
        if u in self.adj and v in self.adj[u]:
            self.adj[u].remove(v)
        if v in self.adj and u in self.adj[v]:
            self.adj[v].remove(u)
        
    def _get_local_radius(self, node_idx):
        neighbors = self.adj.get(node_idx, set())
        if not neighbors:
            return self.base_radius
            
        dists = [np.linalg.norm(self.nodes[node_idx] - self.nodes[n]) for n in neighbors]
        raw_radius = 0.5 * min(dists)
        return float(np.clip(raw_radius, self.base_radius, self.max_radius))

    def is_active(self, vector):
        """Check if a vector falls within the network's bounded Voronoi volume."""
        if not self.nodes:
            return False
        
        vec = np.asarray(vector)
        nodes_mat = np.asarray(self.nodes)
    
        if vec.shape[-1] != nodes_mat.shape[-1]:
            raise ValueError(f"Vector dimension {vec.shape[-1]} does not match node dimension {nodes_mat.shape[-1]}.")

        # Vectorized Euclidean distance calculation across all nodes
        dists = np.linalg.norm(nodes_mat - vec, axis=1)
        n1_idx = int(np.argmin(dists))
        n1_dist = dists[n1_idx]
    
        local_radius = self._get_local_radius(n1_idx)
        return bool(n1_dist <= local_radius)

    def _insert_node(self):
        """Standard GNG node insertion based on accumulated topological error."""
        if not self.errors:
            return
            
        # 1. Find unit q with max error
        q = np.argmax(self.errors)
        
        # 2. Find neighbor f of q with max error
        neighbors = list(self.adj.get(q, set()))
        if not neighbors:
            return
            
        f = neighbors[np.argmax([self.errors[n] for n in neighbors])]
        
        # 3. Insert r halfway between q and f
        r_pos = 0.5 * (self.nodes[q] + self.nodes[f])
        self.nodes.append(r_pos)
        self.errors.append(0.0)
        r = len(self.nodes) - 1
        
        # 4. Remove edge (q, f) and insert (q, r), (f, r)
        self._remove_edge(q, f)
        self._add_edge(q, r, age=0)
        self._add_edge(f, r, age=0)
        
        # 5. Decrease errors of q and f, set error of r
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        self.errors[r] = self.errors[q]

    def tidy(self):
        """Removes old edges and isolated nodes (GNG step 7)."""
        # Remove old edges
        for edge in list(self.edges.keys()):
            if self.edges[edge] > self.max_age:
                del self.edges[edge]
                
        # Find nodes with active edges
        connected = set()
        for u, v in self.edges.keys():
            connected.add(u)
            connected.add(v)
            
        # Filter out isolated vectors
        if len(connected) < len(self.nodes) and len(self.nodes) > 1:
            new_nodes, new_errors = [], []
            idx_map = {}
            for i in range(len(self.nodes)):
                if i in connected:
                    idx_map[i] = len(new_nodes)
                    new_nodes.append(self.nodes[i])
                    new_errors.append(self.errors[i])
                    
            self.nodes = new_nodes
            self.errors = new_errors
            
            # Re-map edges to new indices
            new_edges = {}
            for (u, v), age in self.edges.items():
                if u in idx_map and v in idx_map:
                    new_edges[tuple(sorted((idx_map[u], idx_map[v])))] = age
            self.edges = new_edges

        # Rebuild self.adj cleanly from the active edges
        self.adj = {}
        for u, v in self.edges.keys():
            self.adj.setdefault(u, set()).add(v)
            self.adj.setdefault(v, set()).add(u)

    def update(self, vector, valence: bool = True):
        """
        Standard GNG iteration for positive valence. 
        Negative valence retains a localized repulsion logic to preserve your API.
        """
        vector = np.array(vector)
        
        # Initialization requires at least 2 nodes for standard GNG edges
        if len(self.nodes) < 2:
            if valence:
                self.nodes.append(vector)
                self.errors.append(0.0)
                if len(self.nodes) == 2:
                    self._add_edge(0, 1, age=0)
            return

            
        self.update_count += 1

        # 1. Find the two nearest nodes
        dists = [np.linalg.norm(vector - n) for n in self.nodes]
        sorted_idx = np.argsort(dists)
        s1 = sorted_idx[0]
        s2 = sorted_idx[1]

        if valence:
            # 2. A. Increment ages of all edges connected to s1
            for (u, v) in list(self.edges.keys()):
                if u == s1 or v == s1:
                    self.edges[(u, v)] += 1
                    
            # 3. A. Add squared distance to s1's error
            self.errors[s1] += dists[s1] ** 2
            
            # 4. A. Move s1 and its topological neighbors towards the vector
            self.nodes[s1] += self.eb * (vector - self.nodes[s1])
            
            for (u, v) in self.edges.keys():
                if u == s1: self.nodes[v] += self.en * (vector - self.nodes[v])
                elif v == s1: self.nodes[u] += self.en * (vector - self.nodes[u])
                
            # 5. A. Create or reset edge between s1 and s2
            self._add_edge(s1, s2, age=0)
                 
        # standard GNG do not have this: everything in reverse for negative points within the radius
        elif not valence and dists[s1] <= self._get_local_radius(s1): 
            # 2. B. Decrement ages of all edges connected to s1
            for (u, v) in list(self.edges.keys()):
                if u == s1 or v == s1:
                    self.edges[(u, v)] += - 1
                    
            # 3. B. Substract squared distance to s1's error
            self.errors[s1] += - dists[s1] ** 2
            
            # 4. B. Move s1 and its topological neighbors from the vector
            self.nodes[s1] += - self.eb * (vector - self.nodes[s1])
            
            for (u, v) in self.edges.keys():
                if u == s1: self.nodes[v] += - self.en * (vector - self.nodes[v])
                elif v == s1: self.nodes[u] += - self.en * (vector - self.nodes[u])
                
            # 5. B. Delete edge between s1 and s2
            self._remove_edge(s1, s2)
            
        # 6. Remove old edges and isolated nodes
        self.tidy()
            
        # 7. Insert new node periodically based on maximum error
        if self.update_count % self.lambda_step == 0 and len(self.nodes) < self.max_points:
            self._insert_node()
                
        # 8. Global error decay
        for i in range(len(self.errors)):
            self.errors[i] *= self.d

    def compress(self, merge_ratio=0.5):
        """Compresses redundant states (custom utility adapted for updated lists)."""
        if len(self.nodes) < 3:
            return
            
        merged = True
        while merged:
            merged = False
            for (u, v) in list(self.edges.keys()):
                dist = np.linalg.norm(self.nodes[u] - self.nodes[v])
                threshold = min(self._get_local_radius(u), self._get_local_radius(v)) * merge_ratio

                if dist < threshold:
                    # Merge into midpoint
                    new_pos = (self.nodes[u] + self.nodes[v]) / 2.0
                    self.nodes[u] = new_pos  
                    self.errors[u] = (self.errors[u] + self.errors[v]) / 2.0
                    
                    # Re-route edges
                    for (e1, e2), age in list(self.edges.items()):
                        if v in (e1, e2):
                            other = e1 if e2 == v else e2
                            if other != u:
                                new_edge = tuple(sorted((u, other)))
                                self.edges[new_edge] = age
                            del self.edges[(e1, e2)]
                            
                    # Clean up lists
                    self.nodes.pop(v)
                    self.errors.pop(v)
                    
                    # Re-map edges
                    remapped_edges = {}
                    for (e1, e2), age in self.edges.items():
                        new_e1 = e1 - 1 if e1 > v else e1
                        new_e2 = e2 - 1 if e2 > v else e2
                        remapped_edges[tuple(sorted((new_e1, new_e2)))] = age
                    self.edges = remapped_edges
                    
                    merged = True
                    break