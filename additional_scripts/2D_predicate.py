import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class VoronoiHypersolidGNG:
    def __init__(self, max_points, base_radius=1.0, max_radius=2.0, 
                 learning_rate_b=0.2, learning_rate_n=0.006, 
                 max_edge_age=50, lambda_step=50, alpha=0.5, d=0.995):
        
        self.max_points = max_points
        self.base_radius = base_radius 
        self.max_radius = max_radius 
        
        # Standard GNG Parameters
        self.eb = learning_rate_b         
        self.en = learning_rate_n         
        self.max_age = max_edge_age       
        self.lambda_step = lambda_step    
        self.alpha = alpha                
        self.d = d                        
        
        self.nodes = []                   
        self.errors = []                  
        self.edges = {}                   
        self.update_count = 0
        
    def _get_local_radius(self, node_idx):
        neighbors = []
        for (u, v) in self.edges.keys():
            if u == node_idx: neighbors.append(v)
            elif v == node_idx: neighbors.append(u)
                
        if not neighbors:
            return self.base_radius
            
        dists = [np.linalg.norm(self.nodes[node_idx] - self.nodes[n]) for n in neighbors]
        return min(max(dists), self.max_radius)

    def is_inside(self, point):
        if not self.nodes:
            return False
            
        dists = [np.linalg.norm(point - n) for n in self.nodes]
        n1_idx = np.argmin(dists)
        n1_dist = dists[n1_idx]
        
        local_radius = self._get_local_radius(n1_idx)
        return n1_dist <= local_radius

    def _insert_node(self):
        """Standard GNG node insertion based on accumulated topological error."""
        if not self.errors: return
            
        q = np.argmax(self.errors)
        neighbors = []
        for (u, v) in self.edges.keys():
            if u == q: neighbors.append(v)
            elif v == q: neighbors.append(u)
            
        if not neighbors: return
            
        f = neighbors[np.argmax([self.errors[n] for n in neighbors])]
        
        r_pos = 0.5 * (self.nodes[q] + self.nodes[f])
        self.nodes.append(r_pos)
        self.errors.append(0.0) 
        r = len(self.nodes) - 1
        
        edge_qf = tuple(sorted((q, f)))
        if edge_qf in self.edges:
            del self.edges[edge_qf]
            
        self.edges[tuple(sorted((q, r)))] = 0
        self.edges[tuple(sorted((f, r)))] = 0
        
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        self.errors[r] = self.errors[q]

    def tidy(self):
        """Removes old edges and isolated nodes."""
        for edge in list(self.edges.keys()):
            if self.edges[edge] > self.max_age:
                del self.edges[edge]
                
        connected = set()
        for u, v in self.edges.keys():
            connected.add(u)
            connected.add(v)
            
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
            
            new_edges = {}
            for (u, v), age in self.edges.items():
                if u in idx_map and v in idx_map:
                    new_edges[tuple(sorted((idx_map[u], idx_map[v])))] = age
            self.edges = new_edges

    def update(self, point, valence: bool):
        point = np.array(point)
        
        if len(self.nodes) < 2:
            if valence:
                self.nodes.append(point)
                self.errors.append(0.0)
                if len(self.nodes) == 2:
                    self.edges[(0, 1)] = 0
            return

        if valence:
            self.update_count += 1
            dists = [np.linalg.norm(point - n) for n in self.nodes]
            sorted_idx = np.argsort(dists)
            s1, s2 = sorted_idx[0], sorted_idx[1]
            
            for (u, v) in list(self.edges.keys()):
                if u == s1 or v == s1: self.edges[(u, v)] += 1
                    
            self.errors[s1] += dists[s1] ** 2
            self.nodes[s1] += self.eb * (point - self.nodes[s1])
            
            for (u, v) in self.edges.keys():
                if u == s1: self.nodes[v] += self.en * (point - self.nodes[v])
                elif v == s1: self.nodes[u] += self.en * (point - self.nodes[u])
                
            self.edges[tuple(sorted((s1, s2)))] = 0
            self.tidy()
            
            if self.update_count % self.lambda_step == 0 and len(self.nodes) < self.max_points:
                self._insert_node()
                
            for i in range(len(self.errors)):
                self.errors[i] *= self.d
        else:
            dists = [np.linalg.norm(point - n) for n in self.nodes]
            s1 = np.argmin(dists)
            n1_dist = dists[s1]
            local_radius = self._get_local_radius(s1)
            
            if n1_dist <= local_radius:
                direction = self.nodes[s1] - point
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                    push_dist = (local_radius - n1_dist) + (local_radius * 0.1) 
                    self.nodes[s1] += direction * push_dist

    def prune_internal_nodes(self, vector_sum_threshold=0.35, min_neighbors=None):
        if len(self.nodes) < 4: return
        dim = len(self.nodes[0])
        if min_neighbors is None: min_neighbors = 2 * dim

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
                if len(neighbors) < min_neighbors: continue

                u_pos = self.nodes[i]
                norm_vec_sum = np.zeros(dim)

                for n_idx in neighbors:
                    diff = self.nodes[n_idx] - u_pos
                    norm = np.linalg.norm(diff)
                    if norm > 1e-8:
                        norm_vec_sum += diff / norm

                mean_pull = np.linalg.norm(norm_vec_sum) / len(neighbors)

                if mean_pull < vector_sum_threshold:
                    for idx_a in range(len(neighbors)):
                        for idx_b in range(idx_a + 1, len(neighbors)):
                            na, nb = neighbors[idx_a], neighbors[idx_b]
                            edge = tuple(sorted((na, nb)))
                            if edge not in self.edges: self.edges[edge] = 0

                    for (u, v) in list(self.edges.keys()):
                        if i in (u, v): del self.edges[(u, v)]

                    self.nodes.pop(i)
                    self.errors.pop(i)

                    remapped_edges = {}
                    for (u, v), age in self.edges.items():
                        new_u = u - 1 if u > i else u
                        new_v = v - 1 if v > i else v
                        remapped_edges[tuple(sorted((new_u, new_v)))] = age
                    self.edges = remapped_edges
                    pruned_any = True
                    break 

    def compress(self, merge_ratio=0.5):
        if len(self.nodes) < 3: return
        merged = True
        while merged:
            merged = False
            for (u, v) in list(self.edges.keys()):
                dist = np.linalg.norm(self.nodes[u] - self.nodes[v])
                threshold = min(self._get_local_radius(u), self._get_local_radius(v)) * merge_ratio

                if dist < threshold:
                    new_pos = (self.nodes[u] + self.nodes[v]) / 2.0
                    self.nodes[u] = new_pos  
                    self.errors[u] = (self.errors[u] + self.errors[v]) / 2.0
                    
                    for (e1, e2), age in list(self.edges.items()):
                        if v in (e1, e2):
                            other = e1 if e2 == v else e2
                            if other != u:
                                new_edge = tuple(sorted((u, other)))
                                self.edges[new_edge] = age
                            del self.edges[(e1, e2)]
                            
                    self.nodes.pop(v)
                    self.errors.pop(v)
                    
                    remapped_edges = {}
                    for (e1, e2), age in self.edges.items():
                        new_e1 = e1 - 1 if e1 > v else e1
                        new_e2 = e2 - 1 if e2 > v else e2
                        remapped_edges[tuple(sorted((new_e1, new_e2)))] = age
                    self.edges = remapped_edges
                    merged = True
                    break

    def plot_2d(self, title="Voronoi Hypersolid State", test_points=None, true_circles=None, ax=None):
        if not self.nodes: return
        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            show_plot = True
            
        ax.clear()

        # 0. Sampled Points (Faint green for positive, faint red for negative)
        if test_points:
            pos_pts = [p for p, val in test_points if val]
            neg_pts = [p for p, val in test_points if not val]
            if pos_pts:
                pos_arr = np.array(pos_pts)
                ax.scatter(pos_arr[:, 0], pos_arr[:, 1], c='green', alpha=0.15, s=15, zorder=0)
            if neg_pts:
                neg_arr = np.array(neg_pts)
                ax.scatter(neg_arr[:, 0], neg_arr[:, 1], c='red', alpha=0.15, s=15, zorder=0)
        
        # 1. Target True Solid
        if true_circles is not None:
            for center, rad in true_circles:
                tc = plt.Circle(center, rad, color='black', alpha=0.05, 
                                edgecolor='forestgreen', linestyle='--', linewidth=2, zorder=1)
                ax.add_patch(tc)

        # 2. Learned Voronoi boundaries
        for i, node in enumerate(self.nodes):
            local_radius = self._get_local_radius(i)
            circle = plt.Circle((node[0], node[1]), local_radius, color='skyblue', 
                                alpha=0.2, edgecolor='blue', linewidth=1, zorder=2)
            ax.add_patch(circle)
            
        # 3. GNG edges
        for (u, v) in self.edges.keys():
            if u < len(self.nodes) and v < len(self.nodes):
                n1, n2 = self.nodes[u], self.nodes[v]
                ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color='gray', zorder=3, linewidth=1.5)
            
        # 4. Save points (nodes) -> Changed to deep blue
        nodes_arr = np.array(self.nodes)
        ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='darkblue', marker='o', s=40, zorder=4)
        
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.set_title(f"{title} (Nodes: {len(self.nodes)})")
        ax.grid(True, linestyle='--', alpha=0.5)

        if show_plot:
            plt.tight_layout()
            plt.show()

# ==========================================
# ANIMATION AND TESTING SCRIPT
# ==========================================

# Change the max_points and max_radius in the initialization
hs = VoronoiHypersolidGNG(max_points=150, base_radius=0.25, max_radius=0.8)

# Replace the existing target_shapes list
target_shapes = [
    ((0, 0), 4.0),  # Outer boundary
    ((0, 0), 2.0)   # Inner boundary (hole)
]

# Set up the matplotlib figure for animation
fig, ax = plt.subplots(figsize=(10, 6))

updates_per_frame = 50
total_frames = 150

def animate(frame):
    current_batch_points = []

    for _ in range(updates_per_frame):
        pt = np.random.uniform(-7, 7, 2)
        dist = np.linalg.norm(pt - np.array([0, 0]))
        
        valence = 2.0 < dist < 4.0
        hs.update(pt, valence)
        current_batch_points.append((pt, valence))
        
    # Redraw passing the current batch of points
    hs.plot_2d(title=f"GNG Learning - Frame {frame * updates_per_frame} updates", 
               test_points=current_batch_points,
               true_circles=target_shapes, ax=ax)

# Run animation
print("Starting animation... Please wait.")
ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=50, repeat=False)

plt.show()

# Post-animation processing (Static Plots)
print(f"Nodes before compression: {len(hs.nodes)}")
hs.compress(0.5)
print(f"Nodes after compression: {len(hs.nodes)}")

#hs.prune_internal_nodes(0.5)
#print(f"Nodes after prunning internal nodes: {len(hs.nodes)}")

hs.plot_2d(title="After Compression", true_circles=target_shapes)