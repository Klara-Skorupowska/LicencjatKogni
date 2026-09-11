import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Predicate as GNG (Aligned with Fritzke's Standard GNG Literature + custom negative valence)
class Predicate2D():
    def __init__(self, max_points, base_radius=0.1, max_radius=1.0, 
                 learning_rate_b=0.2, learning_rate_n=0.006, 
                 max_edge_age=5, lambda_step=1, alpha=0.5, d=0.995):
        
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
            if u == node_idx:
                neighbors.append(v)
            elif v == node_idx:
                neighbors.append(u)
                
        if not neighbors:
            return self.base_radius
            
        dists = [np.linalg.norm(self.nodes[node_idx] - self.nodes[n]) for n in neighbors]
        return min(max(dists), self.max_radius)

    def is_active(self, vector):
        if not self.nodes:
            return False
        dists = [np.linalg.norm(vector - n) for n in self.nodes]
        n1_idx = np.argmin(dists)
        n1_dist = dists[n1_idx]
        
        local_radius = self._get_local_radius(n1_idx)
        return n1_dist <= local_radius

    def _insert_node(self):
        if not self.errors:
            return
            
        q = np.argmax(self.errors)
        
        neighbors = []
        for (u, v) in self.edges.keys():
            if u == q: neighbors.append(v)
            elif v == q: neighbors.append(u)
            
        if not neighbors:
            return
            
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

    def update(self, vector, valence: bool = True):
        vector = np.array(vector)
        
        if len(self.nodes) < 2:
            if valence:
                self.nodes.append(vector)
                self.errors.append(0.0)
                if len(self.nodes) == 2:
                    self.edges[(0, 1)] = 0
            return

        self.update_count += 1

        dists = [np.linalg.norm(vector - n) for n in self.nodes]
        sorted_idx = np.argsort(dists)
        s1 = sorted_idx[0]
        s2 = sorted_idx[1]

        if valence:
            for (u, v) in list(self.edges.keys()):
                if u == s1 or v == s1:
                    self.edges[(u, v)] += 1
                    
            self.errors[s1] += dists[s1] ** 2
            
            self.nodes[s1] += self.eb * (vector - self.nodes[s1])
            
            for (u, v) in self.edges.keys():
                if u == s1: self.nodes[v] += self.en * (vector - self.nodes[v])
                elif v == s1: self.nodes[u] += self.en * (vector - self.nodes[u])
                
            self.edges[tuple(sorted((s1, s2)))] = 0
                 
        elif not valence and dists[s1] <= self._get_local_radius(s1): 
            for (u, v) in list(self.edges.keys()):
                if u == s1 or v == s1:
                    self.edges[(u, v)] += - 1
                        
            self.errors[s1] += - dists[s1] ** 2
                
            # Move s1 and its topological neighbors away from the vector
            self.nodes[s1] += - self.eb * (vector - self.nodes[s1])
                
            for (u, v) in self.edges.keys():
                if u == s1: self.nodes[v] += - self.en * (vector - self.nodes[v])
                elif v == s1: self.nodes[u] += - self.en * (vector - self.nodes[u])
                    
            # Delete edge between s1 and s2
            if tuple(sorted((s1, s2))) in self.edges:
                del self.edges[tuple(sorted((s1, s2)))]
            
        self.tidy()
            
        if self.update_count % self.lambda_step == 0 and len(self.nodes) < self.max_points:
            self._insert_node()
                
        for i in range(len(self.errors)):
            self.errors[i] *= self.d

    def compress(self, merge_ratio=0.5):
        if len(self.nodes) < 3:
            return
            
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

        # 0. Sampled Points 
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
                # Changed 'color' to 'facecolor'
                tc = plt.Circle(center, rad, facecolor='black', alpha=0.05, 
                                edgecolor='forestgreen', linestyle='--', linewidth=2, zorder=1)
                ax.add_patch(tc)

        # 2. Learned Voronoi boundaries
        for i, node in enumerate(self.nodes):
            local_radius = self._get_local_radius(i)
            # Changed 'color' to 'facecolor'
            circle = plt.Circle((node[0], node[1]), local_radius, facecolor='skyblue', 
                                alpha=0.2, edgecolor='blue', linewidth=1, zorder=2)
            ax.add_patch(circle)
            
        # 3. GNG edges
        for (u, v) in self.edges.keys():
            if u < len(self.nodes) and v < len(self.nodes):
                n1, n2 = self.nodes[u], self.nodes[v]
                ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color='gray', zorder=3, linewidth=1.5)
            
        # 4. Nodes
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

# Use initialization tuned for the specific negative valence logic
hs = Predicate2D(max_points=150, base_radius=0.1, max_radius=2**(0.5), max_edge_age=50, lambda_step=50)

target_shapes = [
    ((0, 0), 4.0),  
    ((0, 0), 2.0)   
]

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
        
    hs.plot_2d(title=f"GNG Learning - Frame {frame * updates_per_frame} updates", 
               test_points=current_batch_points,
               true_circles=target_shapes, ax=ax)

print("Starting animation... Please wait.")
ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=50, repeat=False)

plt.show()

# Post-animation processing 
print(f"Nodes before compression: {len(hs.nodes)}")
hs.compress(0.5)
print(f"Nodes after compression: {len(hs.nodes)}")

hs.plot_2d(title="After Compression", true_circles=target_shapes)