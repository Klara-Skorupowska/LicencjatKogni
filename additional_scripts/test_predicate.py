import numpy as np
import matplotlib.pyplot as plt

class VoronoiHypersolidGNG:
    def __init__(self, max_points, base_radius=1.0, max_radius=2, learning_rate=0.1, max_edge_age=50):
        self.max_points = max_points
        # base_radius is only used for brand new nodes that don't have neighbors yet
        self.base_radius = base_radius 
        self.max_radius = max_radius # maximum local radius
        self.epsilon = learning_rate      
        self.max_age = max_edge_age       
        
        self.nodes = []                   
        self.edges = {}                   
        
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

    def is_inside(self, point):
        """Check if a point is within the Voronoi hypersolid."""
        if not self.nodes:
            return False
            
        # 1. Find the nearest node (This determines which Voronoi cell the point is in)
        dists = [np.linalg.norm(point - n) for n in self.nodes]
        n1_idx = np.argmin(dists)
        n1_dist = dists[n1_idx]
        
        # 2. Check if the point falls within the local bounded Voronoi extent
        local_radius = self._get_local_radius(n1_idx)
        return n1_dist <= local_radius

    def update(self, point, valence: bool):
        point = np.array(point)
        
        if not self.nodes:
            if valence:
                self.nodes.append(point)
            return

        dists = [np.linalg.norm(point - n) for n in self.nodes]
        sorted_idx = np.argsort(dists)
        n1_idx = sorted_idx[0]
        n1_dist = dists[n1_idx]
        
        # Use the local Voronoi boundary instead of a global radius
        local_radius = self._get_local_radius(n1_idx)
        inside = n1_dist <= local_radius

        # RULE 1: If valence matches current state, do nothing to positions.
        if (valence and inside) or (not valence and not inside):
            if valence and len(sorted_idx) > 1:
                self._update_topology(n1_idx, sorted_idx[1])
            return

        # RULE 2: Mismatch - Modify the hypersolid
        if valence and not inside:
            if len(self.nodes) < self.max_points:
                self.nodes.append(point.copy())
                new_idx = len(self.nodes) - 1
                self.edges[tuple(sorted((n1_idx, new_idx)))] = 0
            else:
                self.nodes[n1_idx] += self.epsilon * (point - self.nodes[n1_idx])
                
        elif not valence and inside:
            # FIX: A False Positive is a boundary error. Shift away immediately 
            # regardless of whether we have reached max_points or not.
            direction = self.nodes[n1_idx] - point
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
                # Push it just outside its local Voronoi boundary
                push_dist = (local_radius - n1_dist) + (local_radius * 0.1) 
                self.nodes[n1_idx] += direction * push_dist
                
                # BONUS: Penalize the topology. If a node claims empty space, 
                # age its edges faster so invalid bridges between clusters break!
                '''
                for edge in list(self.edges.keys()):
                    if n1_idx in edge:
                        self.edges[edge] += 5
                '''

        if valence and len(sorted_idx) > 1:
            self._update_topology(n1_idx, sorted_idx[1])

    def prune_internal_nodes(self, vector_sum_threshold=0.35, min_neighbors=None):
        """
        Removes nodes completely enveloped by neighbors (internal nodes)
        to minimize the graph down to boundary-defining save points.
        
        vector_sum_threshold: Threshold for the norm of the normalized direction sum.
                              Values close to 0 mean uniform surround (internal).
        min_neighbors: Minimum degree required to qualify as internal. 
                       Defaults to (2 * dimensionality) to ensure solid enclosure.
        """
        if len(self.nodes) < 4:
            return

        dim = len(self.nodes[0])
        if min_neighbors is None:
            min_neighbors = 2 * dim

        # Build adjacency mapping
        def build_adj():
            adj = {i: [] for i in range(len(self.nodes))}
            for (u, v) in self.edges.keys():
                adj[u].append(v)
                adj[v].append(u)
            return adj

        pruned_any = True
        while pruned_any:
            pruned_any = False
            adj = build_adj()

            for i in range(len(self.nodes)):
                neighbors = adj[i]
                if len(neighbors) < min_neighbors:
                    continue

                # Compute normalized direction vectors to all neighbors
                u_pos = self.nodes[i]
                norm_vec_sum = np.zeros(dim)

                for n_idx in neighbors:
                    diff = self.nodes[n_idx] - u_pos
                    norm = np.linalg.norm(diff)
                    if norm > 1e-8:
                        norm_vec_sum += diff / norm

                # The average directional pull
                mean_pull = np.linalg.norm(norm_vec_sum) / len(neighbors)

                # If vectors balance out, the node is deep inside the solid
                if mean_pull < vector_sum_threshold:
                    # 1. Bridge all neighbors together so topology remains unbroken
                    for idx_a in range(len(neighbors)):
                        for idx_b in range(idx_a + 1, len(neighbors)):
                            na, nb = neighbors[idx_a], neighbors[idx_b]
                            edge = tuple(sorted((na, nb)))
                            if edge not in self.edges:
                                self.edges[edge] = 0

                    # 2. Remove all existing edges connected to this node
                    for (u, v) in list(self.edges.keys()):
                        if i in (u, v):
                            del self.edges[(u, v)]

                    # 3. Drop the internal node
                    self.nodes.pop(i)

                    # 4. Remap remaining edge indices to match the new nodes list
                    remapped_edges = {}
                    for (u, v), age in self.edges.items():
                        new_u = u - 1 if u > i else u
                        new_v = v - 1 if v > i else v
                        remapped_edges[tuple(sorted((new_u, new_v)))] = age
                    self.edges = remapped_edges

                    pruned_any = True
                    break  # Restart iteration to preserve safe index tracking

    def _update_topology(self, n1, n2):
        for edge in list(self.edges.keys()):
            if n1 in edge:
                self.edges[edge] += 1
                
        new_edge = tuple(sorted((n1, n2)))
        self.edges[new_edge] = 0
        
        for edge in list(self.edges.keys()):
            if self.edges[edge] > self.max_age:
                del self.edges[edge]

    def tidy(self):
        if not self.nodes:
            return
            
        connected_nodes = set()
        for u, v in self.edges.keys():
            connected_nodes.add(u)
            connected_nodes.add(v)
            
        new_nodes = []
        idx_map = {}
        for i, n in enumerate(self.nodes):
            if i in connected_nodes or len(self.nodes) == 1:
                idx_map[i] = len(new_nodes)
                new_nodes.append(n)
                
        self.nodes = new_nodes
        
        new_edges = {}
        for (u, v), age in self.edges.items():
            if u in idx_map and v in idx_map:
                new_edges[tuple(sorted((idx_map[u], idx_map[v])))] = age
        self.edges = new_edges

    def compress(self, merge_ratio=0.5):
        """
        Minimizes node count by collapsing redundant, highly-overlapping Voronoi cells.
        merge_ratio: If distance between neighbors < (local_radius * merge_ratio), merge them.
        """
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

    def plot_2d(self, title="Voronoi Hypersolid State", test_points=None, true_circles=None):
        """
        Visualizes the Voronoi-based hypersolid in 2D using matplotlib.
        Draws dynamic bounded circles representing the local volume of each node.
        """
        if not self.nodes:
            print("No nodes to plot.")
            return
            
        if len(self.nodes[0]) != 2:
            print(f"Cannot plot: Data is {len(self.nodes[0])}D. This method only supports 2D.")
            return

        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 1. Draw the Target "True" Solid (Dashed green outline)
        if true_circles is not None:
            added_label = False
            for center, rad in true_circles:
                tc = plt.Circle(
                    center, 
                    rad, 
                    color='black',
                    alpha = 0.2,
                    edgecolor='forestgreen', 
                    linestyle='--', 
                    linewidth=2.5,
                    zorder=0,              
                    label='Target Solid' if not added_label else None
                )
                ax.add_patch(tc)
                added_label = True

        # 2. Draw the learned Voronoi boundaries (dynamic blue circles)
        added_label = False
        max_current_radius = 0.0 # Track largest radius for axis scaling
        
        for i, node in enumerate(self.nodes):
            local_radius = self._get_local_radius(i)
            max_current_radius = max(max_current_radius, local_radius)
            
            circle = plt.Circle(
                (node[0], node[1]), 
                local_radius, 
                color='skyblue', 
                alpha=0.2,            # slightly more transparent due to overlapping sizes
                edgecolor='blue',
                linewidth=1,
                label='Voronoi Extent' if not added_label else None
            )
            ax.add_patch(circle)
            added_label = True
            
        # 3. Draw the GNG edges (topology)
        for (u, v) in self.edges.keys():
            if u < len(self.nodes) and v < len(self.nodes):
                n1 = self.nodes[u]
                n2 = self.nodes[v]
                ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color='gray', zorder=1, linewidth=1.5)
            
        # 4. Draw the save points (nodes)
        nodes_arr = np.array(self.nodes)
        ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='red', marker='o', s=40, zorder=2, label='Save Points')
        
        # 5. Draw test points if provided
        if test_points is not None and len(test_points) > 0:
            test_points = np.array(test_points)
            inside_pts, outside_pts = [], []
            
            for pt in test_points:
                if self.is_inside(pt):
                    inside_pts.append(pt)
                else:
                    outside_pts.append(pt)
                    
            if inside_pts:
                in_arr = np.array(inside_pts)
                ax.scatter(in_arr[:, 0], in_arr[:, 1], c='green', marker='x', s=60, zorder=3, label='Test: Inside')
            if outside_pts:
                out_arr = np.array(outside_pts)
                ax.scatter(out_arr[:, 0], out_arr[:, 1], c='red', marker='x', s=60, zorder=3, label='Test: Outside')
        
        # Formatting the plot
        ax.set_aspect('equal')
        ax.set_title(f"{title} (Nodes: {len(self.nodes)})")
        
        # Place legend outside to avoid covering data
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Calculate axis limits dynamically
        x_min, x_max = np.min(nodes_arr[:, 0]), np.max(nodes_arr[:, 0])
        y_min, y_max = np.min(nodes_arr[:, 1]), np.max(nodes_arr[:, 1])
        
        if test_points is not None and len(test_points) > 0:
            x_min = min(x_min, np.min(test_points[:, 0]))
            x_max = max(x_max, np.max(test_points[:, 0]))
            y_min = min(y_min, np.min(test_points[:, 1]))
            y_max = max(y_max, np.max(test_points[:, 1]))
            
        # Use the largest calculated local radius for padding
        pad = max(max_current_radius * 1.5, self.base_radius)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()


# Initialize a hypersolid with a max of 50 nodes and a radius of 1.5
hs = VoronoiHypersolidGNG(max_points=100, base_radius=0.25, max_radius=2.0, learning_rate=0.2, max_edge_age=5)

# Training loop
for _ in range(10000):
    # Generate a random 2D point
    pt = np.random.uniform(-7, 7, 2)
    
    # Let's say our target "true" solid is two separate circles (clusters)
    dist_to_c1 = np.linalg.norm(pt - np.array([-5, 0]))
    dist_to_c2 = np.linalg.norm(pt - np.array([5, 0]))
    
    # Valence is True if point is inside either circle
    valence = dist_to_c1 < 2.0 or dist_to_c2 < 2.0
    
    # Update the hypersolid
    hs.update(pt, valence)

# The shapes that define the "true" solid from our training loop
# Format is: [ ((center_x, center_y), radius), ... ]
target_shapes = [
    ((-5, 0), 2.0),
    ((5, 0), 2.0)
]

# Generate some random test points
random_test_points = np.random.uniform(-7, 7, (100, 2))

# Plot!
hs.plot_2d(
    title="before pruning", 
    test_points=random_test_points, 
    true_circles=target_shapes
)

print(f"nodes: {len(hs.nodes)}")
hs.compress(0.5)
print(f"nodes: {len(hs.nodes)}")
#hs.prune_internal_nodes(0.5)
#print(f"nodes: {len(hs.nodes)}")

hs.plot_2d(
    title="after pruning", 
    test_points=random_test_points, 
    true_circles=target_shapes
)
