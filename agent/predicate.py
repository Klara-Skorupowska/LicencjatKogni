import numpy as np

# Predicate as GNG
class Predicate():
    def __init__(self, max_vectors, base_radius=0.1, max_radius=1.0, learning_rate=0.1, max_edge_age=10):
        self.max_vectors = max_vectors
        self.base_radius = base_radius    # Distance threshold to be consider valid around isolated node
        self.max_radius = max_radius      # maximum distance threshold to be consider valid (max local radius)
        self.epsilon = learning_rate      # How much to shift vectors
        self.max_age = max_edge_age       # For clustering/outlier removal
        
        self.nodes = []                   # Holds the valid vectors
        self.edges = {}                   # dict of (node_i, node_j) -> age

        self.prunning_step = 5           # step interval for prunning the network (how many times it should be updated before prunning)
        self.update_count = 0

    def _get_local_radius(self, node_idx):
        """
        Calculates the local bounded Voronoi radius for a node based on 
        the distance to its connected topological neighbors.
        """
        neighbors = []
        for (u, v) in self.edges.keys():
            if u == node_idx:
                neighbors.append(v)
            elif v == node_idx:
                neighbors.append(u)
                
        if not neighbors:
            return self.base_radius
            
        # The maximum distance to a connected neighbor defines the extent 
        # of this node's local volume before it transitions to empty space.
        dists = [np.linalg.norm(self.nodes[node_idx] - self.nodes[n]) for n in neighbors]
        
        # We use max(dists) to ensure the solid covers the space between nodes. 
        # (If you want tighter boundaries, you can multiply this by 0.5 to 0.8)
        return min(max(dists), self.max_radius)

    def _update_topology(self, n1, n2):
        """Standard GNG edge aging to track clusters."""
        # Increment ages of all edges connected to the winner
        for edge in list(self.edges.keys()):
            if n1 in edge:
                self.edges[edge] += 1
                
        # Create or reset edge between the two nearest nodes
        new_edge = tuple(sorted((n1, n2)))
        self.edges[new_edge] = 0
        
        # Remove old edges
        for edge in list(self.edges.keys()):
            if self.edges[edge] > self.max_age:
                del self.edges[edge]

    def is_active(self, vector):
        """Check if a vector is valid."""
        if not self.nodes:
            return False
        # 1. Find the nearest node (This determines which Voronoi cell the vector is in)
        dists = [np.linalg.norm(vector - n) for n in self.nodes]
        n1_idx = np.argmin(dists)
        n1_dist = dists[n1_idx]
        
        # 2. Check if the vector falls within the local bounded Voronoi extent
        local_radius = self._get_local_radius(n1_idx)
        return n1_dist <= local_radius

    def update(self, vector, valence: bool):
        """
        Update the predicate based on a new vector and its valence.
        valence: True (positive, should be valid), False (negative, should be invalid)
        """
        self.update_count += 1
        if self.update_count > self.prunning_step:
            self.update_count = 0
            self.tidy()
            self.compress()

        vector = np.array(vector)
        
        # Initialize if empty
        if not self.nodes:
            if valence:
                self.nodes.append(vector)
            return

        # Find the two nearest nodes 
        dists = [np.linalg.norm(vector - n) for n in self.nodes]
        sorted_idx = np.argsort(dists)
        n1_idx = sorted_idx[0]
        n1_dist = dists[n1_idx]
        
        local_radius = self._get_local_radius(n1_idx)
        active = n1_dist <= local_radius

        # RULE 1: If valence matches current state, nothing happens to positions.
        if (valence and active) or (not valence and not active):
            # (We still update GNG edges to maintain the cluster topology if positive)
            if valence and len(self.nodes) > 1:
                self._update_topology(n1_idx, sorted_idx[1])
            return

        # RULE 2: Mismatch - We need to modify the predicate
        if valence and not active:
            # False Negative: vector should be active but isn't
            if len(self.nodes) < self.max_vectors:
                # Expand predicate: Add new vector
                self.nodes.append(vector.copy())
                new_idx = len(self.nodes) - 1
                self.edges[tuple(sorted((n1_idx, new_idx)))] = 0
            else:
                # Shift nearest vector towards this vector so it active
                self.nodes[n1_idx] += self.epsilon * (vector - self.nodes[n1_idx])
                
        elif not valence and active:
            # False Positive: vector should be inactive but is active
            if len(self.nodes) == self.max_vectors:
                # Shift nearest vector away from the negative vector
                direction = self.nodes[n1_idx] - vector
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                    # Push it just enough so the vector is outside the radius
                    local_radius = self._get_local_radius(n1_idx)
                    push_dist = (local_radius - n1_dist) + (local_radius * 0.1) 
                    self.nodes[n1_idx] += direction * push_dist

        # Update edges if we added positive data
        if valence and len(sorted_idx) > 1:
            self._update_topology(n1_idx, sorted_idx[1])

    def compress(self, merge_ratio = 0.5):
        '''
        compresses reduntant states of the GNG
        merge_ratio: If distance between neighbors < (local_radius * merge_ratio), merge them.
        '''
        if len(self.nodes) < 3:
            return
        merged = True
        while merged:
            merged = False
            for (u, v) in list(self.edges.keys()):
                dist = np.linalg.norm(self.nodes[u] - self.nodes[v])
                threshold = min(self._get_local_radius(u), self._get_local_radius(v)) * merge_ratio

                if dist < threshold:
                    # Merge u and v into their midpoint
                    new_pos = (self.nodes[u] + self.nodes[v]) / 2.0
                    self.nodes[u] = new_pos  # Keep u as the merged node
                    
                    # Re-route all of v's edges to u
                    for (e1, e2), age in list(self.edges.items()):
                        if v in (e1, e2):
                            other = e1 if e2 == v else e2
                            if other != u:
                                new_edge = tuple(sorted((u, other)))
                                self.edges[new_edge] = age
                            del self.edges[(e1, e2)]
                            
                    # Remove node v from the list
                    self.nodes.pop(v)
                    
                    # Re-index all edges to account for the removed index v
                    remapped_edges = {}
                    for (e1, e2), age in self.edges.items():
                        new_e1 = e1 - 1 if e1 > v else e1
                        new_e2 = e2 - 1 if e2 > v else e2
                        remapped_edges[tuple(sorted((new_e1, new_e2)))] = age
                    self.edges = remapped_edges
                    
                    merged = True
                    break  # Restart loop to avoid indexing desync


    def tidy(self):
        """
        Clustering: Achieved naturally because unrelated nodes lose connecting edges.
        Outlier removal: Removes any nodes that have no edges (isolated).
        """
        if not self.nodes:
            return
            
        # Find all nodes that have at least one active edge
        connected_nodes = set()
        for u, v in self.edges.keys():
            connected_nodes.add(u)
            connected_nodes.add(v)
            
        # Filter out isolated vectors (outliers)
        new_nodes = []
        idx_map = {}
        for i, n in enumerate(self.nodes):
            # Keep if connected, or if it's the absolute last vector we have
            if i in connected_nodes or len(self.nodes) == 1:
                idx_map[i] = len(new_nodes)
                new_nodes.append(n)
                
        self.nodes = new_nodes
        
        # Re-map edges to new indices
        new_edges = {}
        for (u, v), age in self.edges.items():
            if u in idx_map and v in idx_map:
                new_edges[tuple(sorted((idx_map[u], idx_map[v])))] = age
        self.edges = new_edges
