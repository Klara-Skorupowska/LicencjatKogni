import os
import ast
import glob
import json
import time
import cv2
import numpy as np
import math
from communicator import Communicator

# =============================================================================
# SHARED RENDERERS & UTILS
# =============================================================================

class SensimotorRenderer:
    """Shared drawing logic for sensimotor and graph visualizations."""
    
    def __init__(self):
        # Defaulting to a clean blue palette (BGR format for OpenCV)
        self.primary_color = (28, 122, 138)       # Deep 
        self.secondary_color = (0, 219, 255)      # Light 
        self.tertiary_color = (219, 250, 255)     # Even Ligher 
        
        self.BG_COLOR = (248, 249, 250)
        self.BORDER_COLOR = (40, 44, 52)
        self.TEXT_MAIN = (30, 30, 30)
        self.ROBOT_BODY = (40, 44, 52)
        self.PANEL_BORDER = (210, 215, 220)
        
        self.neutral_gray = (40, 40, 40)

        self.LIDAR_ANGLES = [17, 50, 90, 150, 210, 270, 310, 343]
        self.INVARIANT_THRESHOLD = 0.05

    def create_base_canvas(self, title, subtitle=None):
        img = np.ones((620, 1000, 3), dtype=np.uint8)
        img[:] = self.BG_COLOR
        cv2.rectangle(img, (0, 0), (999, 619), self.BORDER_COLOR, 3)

        cv2.putText(img, title, (35, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, self.TEXT_MAIN, 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(img, subtitle, (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 70, 20), 1, cv2.LINE_AA)
        
        y_line = 85 if subtitle else 65
        cv2.line(img, (35, y_line), (965, y_line), self.PANEL_BORDER, 1, cv2.LINE_AA)
        return img

    def draw_lidar(self, img, lidar_data, center=(230, 280), max_radius=150, is_live=True):
        cv2.putText(img, "LIDAR", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)

        # Draw background rays
        for ang in self.LIDAR_ANGLES:
            rad = np.deg2rad(ang - 90)
            x_end = int(center[0] + max_radius * np.cos(rad))
            y_end = int(center[1] + max_radius * np.sin(rad))
            cv2.line(img, center, (x_end, y_end), self.secondary_color, 2, cv2.LINE_AA)

        # Draw active readings / invariants
        for i, ang in enumerate(self.LIDAR_ANGLES):
            rad = np.deg2rad(ang - 90)
            
            if is_live:
                val = max(min(float(lidar_data[i]), 0.1), 0.0)
                r_val = (val / 0.1) * max_radius
                x_val = int(center[0] + r_val * np.cos(rad))
                y_val = int(center[1] + r_val * np.sin(rad))
                
                cv2.line(img, center, (x_val, y_val), self.primary_color, 3, cv2.LINE_AA)
                cv2.circle(img, (x_val, y_val), 4, self.primary_color, -1, cv2.LINE_AA)
            else:
                val_min, val_max = lidar_data[i]
                spread = val_max - val_min
                
                r_min = (max(min(val_min, 0.1), 0.0) / 0.1) * max_radius
                r_max = (max(min(val_max, 0.1), 0.0) / 0.1) * max_radius
                
                x_min = int(center[0] + r_min * np.cos(rad))
                y_min = int(center[1] + r_min * np.sin(rad))
                x_max = int(center[0] + r_max * np.cos(rad))
                y_max = int(center[1] + r_max * np.sin(rad))
                
                if spread > self.INVARIANT_THRESHOLD:
                    # Changing variable -> Neutral Gray Line
                    cv2.line(img, (x_min, y_min), (x_max, y_max), self.neutral_gray, 4, cv2.LINE_AA)
                else:
                    # Invariant -> Primary Color Dot at average
                    r_avg = (r_min + r_max) / 2
                    x_avg = int(center[0] + r_avg * np.cos(rad))
                    y_avg = int(center[1] + r_avg * np.sin(rad))
                    cv2.circle(img, (x_avg, y_avg), 5, self.primary_color, -1, cv2.LINE_AA)

        # Robot body
        cv2.circle(img, center, 34, self.ROBOT_BODY, -1, cv2.LINE_AA)
        cv2.circle(img, center, 36, (100, 105, 115), 2, cv2.LINE_AA)
        cv2.line(img, (center[0], center[1] - 34), (center[0], center[1] - 18), (255, 255, 255), 2, cv2.LINE_AA)

    def draw_wheels(self, img, left, right, max_vel, center=(230, 520)):
        cv2.putText(img, "Wheels", (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "L", (165, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "R", (265, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        
        cv2.line(img, (140, center[1]), (290, center[1]), self.PANEL_BORDER, 2, cv2.LINE_AA)

        max_h = 40 
        max_vel = max(max_vel, 0.001)

        l_h = int(np.clip((left / max_vel) * max_h, -max_h, max_h))
        r_h = int(np.clip((right / max_vel) * max_h, -max_h, max_h))

        # Background representing max velocity in secondary color
        cv2.rectangle(img, (150, center[1] - max_h), (190, center[1] + max_h), self.secondary_color, -1)
        cv2.rectangle(img, (250, center[1] - max_h), (290, center[1] + max_h), self.secondary_color, -1)

        # Active velocities in primary color
        cv2.rectangle(img, (150, center[1] - l_h), (190, center[1]), self.primary_color, -1)
        cv2.rectangle(img, (250, center[1] - r_h), (290, center[1]), self.primary_color, -1)

    def draw_grids(self, img, camera_data, is_live=True):
        rows, cols = 120//16, 160//16
        cell_size = 210//rows
        
        rf_start_x, rf_start_y = 520, 120
        c_start_x, c_start_y = 520, 380

        cv2.putText(img, "Retinal Ganglion Cells", (rf_start_x, rf_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "Cone Cells", (c_start_x, c_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)

        for r in range(rows):
            for c in range(cols):
                idx = (r * cols + c) * 5
                
                if is_live:
                    on_val, off_val, red, green, blue = camera_data[idx: idx + 5]
                    on_bool = on_val > 0.5
                    off_bool = off_val > 0.5
                    variant_rf, variant_color = False, False
                else:
                    (on_min, on_max), (off_min, off_max), (r_min, r_max), (g_min, g_max), (b_min, b_max) = camera_data[idx: idx + 5]
                    
                    variant_rf = (on_max - on_min > self.INVARIANT_THRESHOLD) or (off_max - off_min > self.INVARIANT_THRESHOLD)
                    variant_color = (r_max - r_min > self.INVARIANT_THRESHOLD) or (g_max - g_min > self.INVARIANT_THRESHOLD) or (b_max - b_min > self.INVARIANT_THRESHOLD)
                    
                    on_bool = (on_min + on_max) / 2.0 > 0.5
                    off_bool = (off_min + off_max) / 2.0 > 0.5
                    red = (r_min + r_max) / 2.0
                    green = (g_min + g_max) / 2.0
                    blue = (b_min + b_max) / 2.0

                # 1. Receptive Field Logic (On/Off)
                if variant_rf:
                    rf_color = self.neutral_gray
                elif on_bool and not off_bool:
                    rf_color = self.primary_color
                elif off_bool and not on_bool:
                    rf_color = self.secondary_color
                else:
                    rf_color = self.tertiary_color

                rf_x = rf_start_x + (c * cell_size)
                rf_y = rf_start_y + (r * cell_size)
                cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), rf_color, -1)

                # 2. Color Perception Logic (RGB)
                if variant_color:
                    cp_color = self.neutral_gray
                else:
                    if max(red, green, blue) <= 1.0:
                        red, green, blue = red * 255.0, green * 255.0, blue * 255.0
                    cp_color = (int(np.clip(blue, 0, 255)), int(np.clip(green, 0, 255)), int(np.clip(red, 0, 255)))
                
                cp_x = c_start_x + (c * cell_size)
                cp_y = c_start_y + (r * cell_size)
                cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), cp_color, -1)

    def draw_transition_graph(self, img, graph_data):
        """Draws the transition graph directly onto the OpenCV image canvas."""
        nodes = list(graph_data.keys())
        if not nodes:
            cv2.putText(img, "No graph data", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, self.GRID_NEUTRAL, 2)
            return

        center = (500, 350)
        radius = 220
        node_radius = 50
        positions = {}
        
        # Circular Layout
        angle_step = 2 * math.pi / len(nodes)
        for i, node in enumerate(nodes):
            x = int(center[0] + radius * math.cos(i * angle_step))
            y = int(center[1] + radius * math.sin(i * angle_step))
            positions[node] = (x, y)

        # Draw Edges
        for source, targets in graph_data.items():
            if source not in positions: continue
            pt1 = positions[source]
            for target in targets:
                if target not in positions: continue
                pt2 = positions[target]
                
                # Math to draw arrow exactly to the edge of the circle, not center
                dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                dist = math.hypot(dx, dy)
                if dist == 0: continue
                
                start_x = int(pt1[0] + (node_radius * dx / dist))
                start_y = int(pt1[1] + (node_radius * dy / dist))
                end_x = int(pt2[0] - (node_radius * dx / dist))
                end_y = int(pt2[1] - (node_radius * dy / dist))
                
                cv2.arrowedLine(img, (start_x, start_y), (end_x, end_y), self.secondary_color, 2, tipLength=0.05)

        # Draw Nodes
        for node, (x, y) in positions.items():
            cv2.circle(img, (x, y), node_radius, self.secondary_color, -1)
            cv2.circle(img, (x, y), node_radius, self.primary_color, 2)
            
            # Simple text wrap logic
            text_size = cv2.getTextSize(node, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            txt_x = x - text_size[0] // 2
            txt_y = y + text_size[1] // 2
            cv2.putText(img, node, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)


# =============================================================================
# PICTURE GENERATORS (STATIC)
# =============================================================================

class PictureGenerator:
    def __init__(self, logs_root="logs"):
        self.logs_root = logs_root
        self.renderer = SensimotorRenderer()

    def get_latest_session(self):
        if not os.path.exists(self.logs_root): return None
        session_dirs = [os.path.join(self.logs_root, d) for d in os.listdir(self.logs_root) if os.path.isdir(os.path.join(self.logs_root, d))]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def run(self): raise NotImplementedError


class GNGPictureGenerator(PictureGenerator):
    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        symbols_dir = os.path.join(latest_session, "PDDL", "symbols")
        if not os.path.exists(symbols_dir): return
        
        for json_path in glob.glob(os.path.join(symbols_dir, "*.json")):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                nodes = np.array(data.get("nodes", []))
                if len(nodes) == 0: continue
                
                # Find invariants: get min and max across all vectors for this GNG
                min_vals = np.min(nodes, axis=0)
                max_vals = np.max(nodes, axis=0)
                
                file_base = os.path.splitext(os.path.basename(json_path))[0]
                img = self.renderer.create_base_canvas(f"GNG INVARIANTS: {file_base}")
                
                lidar_data = list(zip(min_vals[:8], max_vals[:8]))
                self.renderer.draw_lidar(img, lidar_data, is_live=False)
                
                camera_data = list(zip(min_vals[8:], max_vals[8:]))
                self.renderer.draw_grids(img, camera_data, is_live=False)

                out_dir = os.path.join(symbols_dir, "pictures")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{file_base}_invariants.png")
                cv2.imwrite(out_path, img)
                print(f"[PictureGenerator] Saved GNG invariants to {out_path}")
            except Exception as e:
                print(f"[PictureGenerator] Failed to process {json_path}: {e}")


class TransGraphPictureGenerator(PictureGenerator):
    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        graphs_dir = os.path.join(latest_session, "graphs")

        json_path = os.path.join(graphs_dir, "trans_graph.json")
        if not os.path.exists(json_path): return
        
        try:
            with open(json_path, 'r') as f:
                graph_data = json.load(f)
                
            img = self.renderer.create_base_canvas("TRANSITION GRAPH MONITOR")
            self.renderer.draw_transition_graph(img, graph_data)

            out_dir = os.path.join(graphs_dir, "pictures")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "trans_graph_visualised.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved Transition Graph image to {out_path}")
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate TransGraph: {e}")


class PipelinePictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs"):
        self.generators = [GNGPictureGenerator(logs_root), TransGraphPictureGenerator(logs_root)]

    def run(self):
        for gen in self.generators:
            gen.run()

# =============================================================================
# LIVE MONITORING 
# =============================================================================

class LiveSensimotorMonitor(SensimotorRenderer):
    def __init__(self, bus: Communicator, timeout: float = 0.1):
        super().__init__()
        self.bus = bus
        self.timeout = timeout
        self.window_name = "Live Sensimotor Monitor"
        self._window_created = False

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_created = True

    def _safe_call(self, topic: str):
        try:
            try: return self.bus.call_service(topic, timeout=self.timeout)
            except TypeError: return self.bus.call_service(topic)
        except Exception: return None

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        lidar_readings = []
        for direction in self.LIDAR_ANGLES:
            val = self._safe_call(f"/lidar/{direction}/ask/value")
            if val is None: return
            lidar_readings.append(val)

        raw_coded = self._safe_call("/camera/ask/coded")
        if raw_coded is None: return
        cam_array = np.asarray(raw_coded, dtype=float).flatten()
        if len(cam_array) < (6 * 8 * 5): return

        speeds = self._safe_call("/wheels/ask/speed")
        max_vel = self._safe_call("/get/wheels/max_velocity")
        if speeds is None or max_vel is None: return

        img = self.create_base_canvas("LIVE SENSIMOTOR MONITOR")
        self.draw_lidar(img, lidar_readings, is_live=True)
        self.draw_wheels(img, speeds[0], speeds[1], float(max_vel))
        self.draw_grids(img, cam_array, is_live=True)

        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


class LiveGraphMonitor(SensimotorRenderer):
    def __init__(self, logs_root="logs", trans_filename="trans_graph.json", poll_interval=1.0):
        super().__init__()
        self.logs_root = logs_root
        self.trans_filename = trans_filename
        self.poll_interval = poll_interval
        self._last_poll_time = 0.0
        self._last_trans_mtime = 0.0
        self.trans_window = "Live Transition Graph"
        self._window_created = False

    def get_latest_session(self):
        if not os.path.exists(self.logs_root): return None
        session_dirs = [os.path.join(self.logs_root, d) for d in os.listdir(self.logs_root) if os.path.isdir(os.path.join(self.logs_root, d))]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.trans_window, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("TRANSITION GRAPH MONITOR", "Waiting for data...")
            cv2.imshow(self.trans_window, placeholder)
            self._window_created = True

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        now = time.time()
        if now - self._last_poll_time < self.poll_interval: return
        self._last_poll_time = now

        latest_session = self.get_latest_session()
        if not latest_session: return

        trans_path = os.path.join(latest_session, "graphs", self.trans_filename)

        try:
            if os.path.exists(trans_path) and os.path.getsize(trans_path) > 0:
                mtime = os.path.getmtime(trans_path)
                if mtime > self._last_trans_mtime:
                    with open(trans_path, "r") as f:
                        graph_data = json.load(f)
                        
                    img = self.create_base_canvas("LIVE TRANSITION GRAPH")
                    self.draw_transition_graph(img, graph_data)
                    cv2.imshow(self.trans_window, img)
                    
                    self._last_trans_mtime = mtime
        except Exception as e:
            print(f"Error loading live graph: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.trans_window)
            self._window_created = False

import networkx as nx

class LiveGNGMonitor(SensimotorRenderer):
    def __init__(self, logs_root="logs", poll_interval=1.0):
        super().__init__()
        self.logs_root = logs_root
        self.poll_interval = poll_interval
        self._last_poll_time = 0.0
        
        self.window_name = "Live GNG Plane"
        self._window_created = False
        
        # Track modification times to avoid redundant rendering
        self._last_mtimes = {}
        
        # Dynamic color tracking for predicates
        self.gng_colors = {}
        # A list of distinct, vibrant BGR colors
        self.color_palette = [
            (60, 60, 220),   # Red-ish
            (60, 220, 60),   # Green-ish
            (220, 100, 60),  # Blue-ish
            (60, 200, 220),  # Yellow-ish
            (200, 60, 200),  # Magenta-ish
            (220, 200, 60),  # Cyan-ish
            (100, 120, 255), # Light Red
            (255, 120, 100), # Light Blue
            (100, 255, 120)  # Light Green
        ]

    def get_latest_session(self):
        if not os.path.exists(self.logs_root): return None
        session_dirs = [os.path.join(self.logs_root, d) for d in os.listdir(self.logs_root) if os.path.isdir(os.path.join(self.logs_root, d))]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("LIVE GNG PLANE", "Waiting for predicate data...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def _get_color(self, predicate_name):
        if predicate_name not in self.gng_colors:
            color_idx = len(self.gng_colors) % len(self.color_palette)
            self.gng_colors[predicate_name] = self.color_palette[color_idx]
        return self.gng_colors[predicate_name]

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        now = time.time()
        if now - self._last_poll_time < self.poll_interval: return
        self._last_poll_time = now

        latest_session = self.get_latest_session()
        if not latest_session: return

        symbols_dir = os.path.join(latest_session, "PDDL", "symbols")
        if not os.path.exists(symbols_dir): return

        json_files = glob.glob(os.path.join(symbols_dir, "*.json"))
        if not json_files: return

        # 1. Check if any file was updated
        needs_update = False
        for f in json_files:
            mtime = os.path.getmtime(f)
            if self._last_mtimes.get(f, 0.0) < mtime:
                needs_update = True
                self._last_mtimes[f] = mtime
                
        if not needs_update: return

        # 2. Build the unified topological graph
        G = nx.Graph()
        node_colors = {}
        edge_colors = {}
        
        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                pred_name = os.path.splitext(os.path.basename(json_path))[0]
                color = self._get_color(pred_name)
                
                # Add nodes (use string names to prevent ID clashes between files)
                nodes = data.get("nodes", [])
                for i in range(len(nodes)):
                    node_id = f"{pred_name}_{i}"
                    G.add_node(node_id)
                    node_colors[node_id] = color

                # Add edges
                edges_dict = data.get("edges", {})
                for edge_str in edges_dict.keys():
                    u_str, v_str = edge_str.split(",")
                    u_id = f"{pred_name}_{u_str}"
                    v_id = f"{pred_name}_{v_str}"
                    G.add_edge(u_id, v_id)
                    edge_colors[(u_id, v_id)] = color
                    edge_colors[(v_id, u_id)] = color # undirected
            except Exception as e:
                print(f"[LiveGNGMonitor] Failed to read {json_path}: {e}")

        if len(G.nodes) == 0: return

        # 3. Calculate 2D Projection using Spring Layout
        pos = nx.spring_layout(G, seed=42, k=0.25) # k controls optimal distance between nodes

        # 4. Render to OpenCV Canvas
        img = self.create_base_canvas("LIVE GNG PLANE", "Topological state space mapping")
        
        # Scaling variables to fit the canvas
        center_x, center_y = 500, 340
        scale = 230 

        # Draw Edges
        for u, v in G.edges():
            pt1 = (int(center_x + pos[u][0] * scale), int(center_y + pos[u][1] * scale))
            pt2 = (int(center_x + pos[v][0] * scale), int(center_y + pos[v][1] * scale))
            color = edge_colors.get((u, v), self.neutral_gray)
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)

        # Draw Nodes
        for node in G.nodes():
            pt = (int(center_x + pos[node][0] * scale), int(center_y + pos[node][1] * scale))
            color = node_colors.get(node, self.neutral_gray)
            cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, self.TEXT_MAIN, 1, cv2.LINE_AA) # Dark border

        # 5. Draw Legend
        legend_start_y = 120
        cv2.putText(img, "Predicates:", (820, legend_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)
        
        for i, (pred_name, color) in enumerate(self.gng_colors.items()):
            y_pos = legend_start_y + 25 + (i * 20)
            cv2.circle(img, (830, y_pos - 4), 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, pred_name[:18], (845, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_MAIN, 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

if __name__ == "__main__":
    runner = PipelinePictureGenerator(logs_root="logs")
    runner.run()