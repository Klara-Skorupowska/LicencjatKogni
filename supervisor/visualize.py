import os
import glob
import json
import time
import re
import cv2
import numpy as np
import math
from communicator import Communicator
import networkx as nx
import textwrap

# =============================================================================
# SHARED RENDERERS & UTILS
# =============================================================================

class Renderer:
    """Shared drawing logic for sensimotor and graph visualizations."""
    
    def __init__(self):
        self.primary_color = (195, 105, 30)       # Rich Steel / Slate Blue
        self.secondary_color = (235, 185, 90)     # Sky Blue / Cyan Accent
        self.tertiary_color = (248, 235, 215)     # Pale Ice Blue tint
        
        self.BG_COLOR = (248, 249, 250)
        self.BORDER_COLOR = (40, 44, 52)
        self.TEXT_MAIN = (30, 30, 30)
        self.ROBOT_BODY = (40, 44, 52)
        self.PANEL_BORDER = (210, 215, 220)
        
        self.neutral_gray = (130, 135, 140)
        self.LIDAR_ANGLES = [17, 50, 90, 150, 210, 270, 310, 343]
        self.INVARIANT_THRESHOLD = 0.005

    def compute_hue_stats(self, r_vals, g_vals, b_vals):
        max_v = max(np.max(r_vals), np.max(g_vals), np.max(b_vals))
        if max_v > 1.0:
            r = np.array(r_vals, dtype=np.float32) / 255.0
            g = np.array(g_vals, dtype=np.float32) / 255.0
            b = np.array(b_vals, dtype=np.float32) / 255.0
        else:
            r = np.array(r_vals, dtype=np.float32)
            g = np.array(g_vals, dtype=np.float32)
            b = np.array(b_vals, dtype=np.float32)

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        h = np.zeros_like(r)
        nonzero = delta > 1e-6

        mask = nonzero & (cmax == r)
        h[mask] = (60.0 * (((g[mask] - b[mask]) / delta[mask]) % 6))

        mask = nonzero & (cmax == g)
        h[mask] = (60.0 * (((b[mask] - r[mask]) / delta[mask]) + 2))

        mask = nonzero & (cmax == b)
        h[mask] = (60.0 * (((r[mask] - g[mask]) / delta[mask]) + 4))

        angles = np.deg2rad(h)
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        R = np.hypot(sin_mean, cos_mean)
        circ_var = 1.0 - R

        mean_angle_deg = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0

        hsv_pixel = np.uint8([[[int(mean_angle_deg / 2.0), 255, 255]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        mean_bgr = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
        return mean_bgr, circ_var

    def draw_sample_badge(self, img, n_points, position=(830, 48)):
        text = f"N = {n_points} samples"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = position
        pad_x, pad_y = 10, 6
        
        cv2.rectangle(img, (x, y - text_h - pad_y), (x + text_w + 2 * pad_x, y + baseline + pad_y), self.tertiary_color, -1)
        cv2.rectangle(img, (x, y - text_h - pad_y), (x + text_w + 2 * pad_x, y + baseline + pad_y), self.primary_color, 1)
        cv2.putText(img, text, (x + pad_x, y), font, scale, self.TEXT_MAIN, thickness, cv2.LINE_AA)

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

        for ang in self.LIDAR_ANGLES:
            rad = np.deg2rad(ang - 90)
            x_end = int(center[0] + max_radius * np.cos(rad))
            y_end = int(center[1] + max_radius * np.sin(rad))
            cv2.line(img, center, (x_end, y_end), self.tertiary_color, 2, cv2.LINE_AA)

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
                val_min, val_max, val_mean = lidar_data[i]
                r_min = (max(min(val_min, 1), 0.0) / 1) * max_radius
                r_max = (max(min(val_max, 1), 0.0) / 1) * max_radius
                r_mean = (max(min(val_mean, 1), 0.0) / 1) * max_radius
                
                x_min = int(center[0] + r_min * np.cos(rad))
                y_min = int(center[1] + r_min * np.sin(rad))
                x_max = int(center[0] + r_max * np.cos(rad))
                y_max = int(center[1] + r_max * np.sin(rad))
                x_mean = int(center[0] + r_mean * np.cos(rad))
                y_mean = int(center[1] + r_mean * np.sin(rad))
                
                cv2.line(img, (x_min, y_min), (x_max, y_max), self.secondary_color, 4, cv2.LINE_AA)
                cv2.circle(img, (x_mean, y_mean), 5, self.primary_color, -1, cv2.LINE_AA)

        cv2.circle(img, center, 34, self.ROBOT_BODY, -1, cv2.LINE_AA)
        cv2.circle(img, center, 36, (100, 105, 115), 2, cv2.LINE_AA)
        cv2.line(img, (center[0], center[1] - 34), (center[0], center[1] - 18), (255, 255, 255), 2, cv2.LINE_AA)

    def draw_grids(self, img, camera_data, is_live=True):
        rows, cols = 120 // 20, 160 // 20
        cell_size = 210 // rows
        
        rf_start_x, rf_start_y = 520, 120
        c_start_x, c_start_y = 520, 380

        cv2.putText(img, "Retinal Ganglion Cells", (rf_start_x, rf_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "Cone Cells", (c_start_x, c_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)

        for r in range(rows):
            for c in range(cols):
                idx = (r * cols + c) * 5
                rf_x = rf_start_x + (c * cell_size)
                rf_y = rf_start_y + (r * cell_size)
                cp_x = c_start_x + (c * cell_size)
                cp_y = c_start_y + (r * cell_size)

                if is_live:
                    on_val, off_val, red, green, blue = camera_data[idx: idx + 5]
                    if on_val > 0.5 and off_val <= 0.5:
                        rf_color = self.primary_color
                    elif off_val > 0.5 and on_val <= 0.5:
                        rf_color = self.secondary_color
                    else:
                        rf_color = self.tertiary_color
                    
                    if max(red, green, blue) <= 1.0:
                        red, green, blue = red * 255.0, green * 255.0, blue * 255.0
                    cp_color = (int(np.clip(blue, 0, 255)), int(np.clip(green, 0, 255)), int(np.clip(red, 0, 255)))
                    
                    cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), rf_color, -1)
                    cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), cp_color, -1)
                else:
                    ganglion_stats, cone_hue_stats = camera_data
                    cell_num = r * cols + c
                    on_stat = ganglion_stats[idx]
                    off_stat = ganglion_stats[idx + 1]

                    if on_stat[1] <= self.INVARIANT_THRESHOLD and off_stat[1] <= self.INVARIANT_THRESHOLD:
                        on_bool = on_stat[0] > 0.5
                        off_bool = off_stat[0] > 0.5
                        if on_bool and not off_bool:
                            rf_color = self.primary_color
                        elif off_bool and not on_bool:
                            rf_color = self.secondary_color
                        else:
                            rf_color = self.tertiary_color
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), rf_color, -1)
                    else:
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), self.PANEL_BORDER, 1)

                    mean_bgr, hue_circ_var = cone_hue_stats[cell_num]
                    if hue_circ_var <= self.INVARIANT_THRESHOLD:
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), mean_bgr, -1)
                    else:
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), self.PANEL_BORDER, 1)

    def draw_action_graph(self, img, graph_data):
        """Draws possible action transitions extracted from PDDL operators."""
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            cv2.putText(img, "No PDDL actions or moves available", (260, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.neutral_gray, 2)
            return

        center = (500, 350)
        node_radius = 55
        
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for u, v in edges:
            if u in nodes and v in nodes:
                G.add_edge(u, v)

        optimal_dist = node_radius / math.sqrt(max(len(nodes), 1))
        raw_pos = nx.spring_layout(G, k=optimal_dist, center=center, scale=210, seed=42)
        positions = {node: (int(coords[0]), int(coords[1])) for node, coords in raw_pos.items()}

        for u, v in edges:
            if u not in positions or v not in positions:
                continue

            pt1 = positions[u]
            pt2 = positions[v]

            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue

            start_x = int(pt1[0] + (node_radius * dx / dist))
            start_y = int(pt1[1] + (node_radius * dy / dist))
            end_x = int(pt2[0] - (node_radius * dx / dist))
            end_y = int(pt2[1] - (node_radius * dy / dist))

            P1 = np.array([start_x, start_y], dtype=float)
            P2 = np.array([end_x, end_y], dtype=float)
            gap_dx, gap_dy = P2[0] - P1[0], P2[1] - P1[1]
            gap_dist = math.hypot(gap_dx, gap_dy)

            if gap_dist > 0:
                M = (P1 + P2) / 2.0
                nx_v = -gap_dy / gap_dist
                ny_v = gap_dx / gap_dist
                curve_offset = gap_dist * 0.18
                C = M + np.array([nx_v, ny_v]) * curve_offset

                t = np.linspace(0, 1, 20).reshape(-1, 1)
                curve_pts = ((1 - t)**2 * P1 + 2 * (1 - t) * t * C + t**2 * P2).astype(np.int32)
                
                cv2.polylines(img, [curve_pts], False, self.primary_color, 2, cv2.LINE_AA)

                tx, ty = P2[0] - C[0], P2[1] - C[1]
                t_len = math.hypot(tx, ty)
                if t_len > 0:
                    tx /= t_len
                    ty /= t_len
                    arrow_start = (int(P2[0] - tx * 18), int(P2[1] - ty * 18))
                    cv2.arrowedLine(img, arrow_start, (int(P2[0]), int(P2[1])),
                                    self.primary_color, 2, tipLength=0.4, line_type=cv2.LINE_AA)

        for node, (x, y) in positions.items():
            cv2.circle(img, (x, y), node_radius, self.tertiary_color, -1)
            cv2.circle(img, (x, y), node_radius, self.primary_color, 2)
            display_name = node.replace('_', ' ')
            wrapped_lines = textwrap.wrap(display_name, width=12)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            line_metrics = [cv2.getTextSize(line, font, font_scale, thickness) for line in wrapped_lines]
            total_height = sum(th + baseline for (tw, th), baseline in line_metrics) + max(0, len(wrapped_lines) - 1) * 3
            current_y = y - total_height // 2

            for line, ((tw, th), baseline) in zip(wrapped_lines, line_metrics):
                txt_x = int(x - tw / 2)
                txt_y = int(current_y + th)
                cv2.putText(img, line, (txt_x, txt_y), font, font_scale, self.TEXT_MAIN, thickness, cv2.LINE_AA)
                current_y += th + baseline + 3

    def draw_unified_GNG(self, img, G, node_colors, edge_colors, gng_colors, legend_title="Predicates:"):
        """Draws a topological GNG plane onto canvas."""
        if len(G.nodes) == 0:
            return

        pos = nx.spring_layout(G, seed=42, k=0.25)
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        raw_w = max_x - min_x if max_x > min_x else 1.0
        raw_h = max_y - min_y if max_y > min_y else 1.0

        max_r_px = max([max(int(G.nodes[n].get("radius", 0.0) * 35), 5) for n in G.nodes()], default=5)
        pad_left = max_r_px + 30
        pad_right = max_r_px + 220
        pad_y_top = max_r_px + 100
        pad_y_bot = max_r_px + 30
        
        avail_w = 1000 - (pad_left + pad_right)
        avail_h = 620 - (pad_y_top + pad_y_bot)
        scale = min(avail_w / raw_w, avail_h / raw_h)
        center_x = pad_left + avail_w / 2 - ((min_x + max_x) / 2) * scale
        center_y = pad_y_top + avail_h / 2 - ((min_y + max_y) / 2) * scale

        # 1. Local radii
        for node in G.nodes():
            pt = (int(center_x + pos[node][0] * scale), int(center_y + pos[node][1] * scale))
            base_color = node_colors.get(node, self.neutral_gray)
            light_color = tuple(int(0.75 * bg + 0.25 * bc) for bg, bc in zip(self.BG_COLOR, base_color))
            r_val = G.nodes[node].get("radius", 0.0)
            if r_val > 0:
                r_px = max(int(r_val * 35), 5) 
                cv2.circle(img, pt, r_px, light_color, -1, cv2.LINE_AA)

        # 2. Edges
        for u, v in G.edges():
            pt1 = (int(center_x + pos[u][0] * scale), int(center_y + pos[u][1] * scale))
            pt2 = (int(center_x + pos[v][0] * scale), int(center_y + pos[v][1] * scale))
            color = edge_colors.get((u, v), self.neutral_gray)
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)

        # 3. Nodes
        for node in G.nodes():
            pt = (int(center_x + pos[node][0] * scale), int(center_y + pos[node][1] * scale))
            color = node_colors.get(node, self.neutral_gray)
            cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, self.TEXT_MAIN, 1, cv2.LINE_AA)

        # 4. Legend
        legend_start_y = 120
        cv2.putText(img, legend_title, (800, legend_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for i, (pred_name, color) in enumerate(gng_colors.items()):
            y_pos = legend_start_y + 25 + (i * 20)
            if y_pos > 600:
                break
            cv2.circle(img, (810, y_pos - 4), 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, pred_name[:20], (825, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)


# =============================================================================
# LIVE MONITORS
# =============================================================================

class BaseLiveGNGPlane(Renderer):
    """Abstract base monitor for rendering a plane of GNGs from a target directory."""
    def __init__(self, logs_root="logs", poll_interval=1.0):
        super().__init__()
        self.logs_root = logs_root
        self.poll_interval = poll_interval
        self._last_poll_time = 0.0
        self._window_created = False
        self._last_mtimes = {}
        self.gng_colors = {}
        self.color_palette = [
            (60, 60, 220), (60, 220, 60), (220, 100, 60),
            (60, 200, 220), (200, 60, 200), (220, 200, 60),
            (100, 120, 255), (255, 120, 100), (100, 255, 120)
        ]

    def get_latest_session(self):
        if not os.path.exists(self.logs_root): return None
        session_dirs = [os.path.join(self.logs_root, d) for d in os.listdir(self.logs_root) if os.path.isdir(os.path.join(self.logs_root, d))]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def _get_color(self, name):
        if name not in self.gng_colors:
            idx = len(self.gng_colors) % len(self.color_palette)
            self.gng_colors[name] = self.color_palette[idx]
        return self.gng_colors[name]

    def get_target_files(self, latest_session):
        raise NotImplementedError

    def get_titles(self):
        return "LIVE GNG PLANE", "GNG Topologies", "Predicates:"

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        now = time.time()
        if now - self._last_poll_time < self.poll_interval: return
        self._last_poll_time = now

        latest_session = self.get_latest_session()
        if not latest_session: return

        json_files = self.get_target_files(latest_session)
        if not json_files: return

        needs_update = False
        for f in json_files:
            try:
                mtime = os.path.getmtime(f)
                if self._last_mtimes.get(f, 0.0) < mtime:
                    needs_update = True
                    self._last_mtimes[f] = mtime
            except OSError:
                pass

        if not needs_update and len(self._last_mtimes) > 0:
            return

        G = nx.Graph()
        node_colors = {}
        edge_colors = {}

        for json_path in json_files:
            try:
                with open(json_path, 'r') as fp:
                    data = json.load(fp)
                pred_name = os.path.splitext(os.path.basename(json_path))[0]
                color = self._get_color(pred_name)
                nodes = data.get("nodes", [])
                radiuses = data.get("local_radiuses", [])

                for i in range(len(nodes)):
                    node_id = f"{pred_name}_{i}"
                    r_val = radiuses[i] if i < len(radiuses) else 0.0
                    G.add_node(node_id, radius=r_val)
                    node_colors[node_id] = color

                edges_dict = data.get("edges", {})
                for edge_str in edges_dict.keys():
                    u_str, v_str = edge_str.split(",")
                    u_id = f"{pred_name}_{u_str}"
                    v_id = f"{pred_name}_{v_str}"
                    G.add_edge(u_id, v_id)
                    edge_colors[(u_id, v_id)] = color
                    edge_colors[(v_id, u_id)] = color
            except Exception:
                continue

        if len(G.nodes) == 0: return
        title, sub, legend = self.get_titles()
        img = self.create_base_canvas(title, sub)
        self.draw_unified_GNG(img, G, node_colors, edge_colors, self.gng_colors, legend_title=legend)
        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


class LiveEffectsPlaneMonitor(BaseLiveGNGPlane):
    """Plane 1: Visualizes all Action Effect Sets (Eff_o)."""
    def __init__(self, logs_root="logs", poll_interval=1.0):
        super().__init__(logs_root=logs_root, poll_interval=poll_interval)
        self.window_name = "Live Effects Plane"

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("LIVE EFFECTS PLANE", "Waiting for effect GNGs...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def get_target_files(self, latest_session):
        gng_dir = os.path.join(latest_session, "GNG")
        if not os.path.exists(gng_dir): return []
        return glob.glob(os.path.join(gng_dir, "*_effect.json"))

    def get_titles(self):
        return "LIVE EFFECTS PLANE", "Topological mapping of option effect regions", "Action Effects:"


class LiveSymbolsPlaneMonitor(BaseLiveGNGPlane):
    """Plane 2: Visualizes all Grounded Abstract Symbols (Subspace Intersections)."""
    def __init__(self, logs_root="logs", poll_interval=1.0):
        super().__init__(logs_root=logs_root, poll_interval=poll_interval)
        self.window_name = "Live Symbols Plane"

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("LIVE SYMBOLS PLANE", "Waiting for discovered symbols...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def get_target_files(self, latest_session):
        symbols_dir = os.path.join(latest_session, "PDDL", "symbols")
        if not os.path.exists(symbols_dir): return []
        return glob.glob(os.path.join(symbols_dir, "*.json"))

    def get_titles(self):
        return "LIVE SYMBOLS PLANE", "Discovered propositional symbol preconditions", "Symbols:"


class LiveActionInitMonitor(Renderer):
    """Displays the continuous initiation set (I_o) of the currently active action."""
    def __init__(self, bus: Communicator, logs_root="logs", poll_interval=0.5):
        super().__init__()
        self.bus = bus
        self.logs_root = logs_root
        self.poll_interval = poll_interval
        self._last_poll_time = 0.0
        self.window_name = "Current Action Initiation Set"
        self._window_created = False
        self._last_action = None

    def get_latest_session(self):
        if not os.path.exists(self.logs_root): return None
        session_dirs = [os.path.join(self.logs_root, d) for d in os.listdir(self.logs_root) if os.path.isdir(os.path.join(self.logs_root, d))]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("ACTIVE ACTION INITIATION", "Waiting for running action...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def _resolve_action_name(self):
        try:
            res = self.bus.call_service("agent/ask/action")
            if isinstance(res, dict):
                return res.get("action") or res.get("name")
            return res
        except Exception:
            return None

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        now = time.time()
        if now - self._last_poll_time < self.poll_interval: return
        self._last_poll_time = now

        action_name = self._resolve_action_name()
        if not action_name:
            return

        latest_session = self.get_latest_session()
        if not latest_session:
            return

        init_file = os.path.join(latest_session, "GNG", f"{action_name}_init.json")
        if not os.path.isfile(init_file):
            return

        try:
            with open(init_file, "r") as f:
                data = json.load(f)

            nodes = np.array(data.get("nodes", []))
            if len(nodes) == 0:
                return

            # Compute medoid representative of the initiation set
            centroid = np.mean(nodes, axis=0)
            dists = np.linalg.norm(nodes - centroid, axis=1)
            rep_idx = int(np.argmin(dists))
            rep_node = nodes[rep_idx]

            display_title = str(action_name).replace("_", " ").upper()
            img = self.create_base_canvas(
                f"CURRENT ACTION PRECONDITION: {display_title}",
                f"Initiation Set Medoid Prototype ({len(nodes)} total nodes)",
            )

            # Dimensions 0:8 LIDAR, 8:248 Ganglion + Cone Cells
            self.draw_lidar(img, rep_node[:8], is_live=True)
            self.draw_grids(img, rep_node[8:248], is_live=True)
            self.draw_sample_badge(img, n_points=len(nodes))

            cv2.imshow(self.window_name, img)
        except Exception as e:
            print(f"[LiveActionInitMonitor] Error rendering: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


class LivePDDLMovesMonitor(Renderer):
    """Parses newest domain.pddl and displays possible action transitions."""
    def __init__(self, logs_root="logs", poll_interval=1.0):
        super().__init__()
        self.logs_root = logs_root
        self.poll_interval = poll_interval
        self._last_poll_time = 0.0
        self._last_domain_mtime = 0.0
        self.window_name = "Possible Moves (PDDL)"
        self._window_created = False

    def get_latest_session(self):
        if not os.path.exists(self.logs_root): return None
        session_dirs = [os.path.join(self.logs_root, d) for d in os.listdir(self.logs_root) if os.path.isdir(os.path.join(self.logs_root, d))]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("PDDL POSSIBLE MOVES", "Waiting for newest domain.pddl...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def _parse_pddl_moves(self, domain_path):
        with open(domain_path, "r") as f:
            content = f.read()

        # Find all (:action <name> ... :effect (and ...)) blocks
        action_blocks = re.findall(r"\(:action\s+([^\s\)]+)(.*?)\s*\)\s*(?=\(:action|\)\s*\Z)", content, re.DOTALL)
        nodes = []
        edges = []

        for act_name, body in action_blocks:
            nodes.append(act_name)
            # Find all (can_run_<target>) in add effect
            effect_match = re.search(r":effect\s+\(and(.*?)\)", body, re.DOTALL)
            if effect_match:
                effect_str = effect_match.group(1)
                # Ignore deleted conditions: filter out (not ...)
                positive_effects = re.sub(r"\(not\s+\([^\)]+\)\)", "", effect_str)
                targets = re.findall(r"\(can_run_([^\s\)]+)\)", positive_effects)
                for tgt in targets:
                    edges.append((act_name, tgt))

        return {"nodes": nodes, "edges": edges}

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        now = time.time()
        if now - self._last_poll_time < self.poll_interval: return
        self._last_poll_time = now

        latest_session = self.get_latest_session()
        if not latest_session: return

        domain_path = os.path.join(latest_session, "PDDL", "newest", "domain.pddl")
        if not os.path.exists(domain_path): return

        try:
            mtime = os.path.getmtime(domain_path)
            if mtime > self._last_domain_mtime:
                graph_data = self._parse_pddl_moves(domain_path)
                img = self.create_base_canvas("PDDL POSSIBLE MOVES", f"Parsed from {domain_path}")
                self.draw_action_graph(img, graph_data)
                cv2.imshow(self.window_name, img)
                self._last_domain_mtime = mtime
        except Exception as e:
            print(f"[LivePDDLMovesMonitor] Parsing error: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False



# =============================================================================
# PICTURE GENERATORS FOR NEW COMPONENTS
# =============================================================================

class PictureGenerator:
    def __init__(self, logs_root="logs", session_dir=None):
        self.logs_root = logs_root
        self.session_dir = session_dir
        self.renderer = Renderer()

    def get_latest_session(self):
        if self.session_dir:
            if os.path.isabs(self.session_dir) or os.path.exists(self.session_dir):
                return self.session_dir
            candidate = os.path.join(self.logs_root, self.session_dir)
            if os.path.exists(candidate):
                return candidate

        if not os.path.exists(self.logs_root): return None
        session_dirs = [os.path.join(self.logs_root, d) for d in os.listdir(self.logs_root) if os.path.isdir(os.path.join(self.logs_root, d))]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def get_pictures_dir(self):
        latest_session = self.get_latest_session()
        if not latest_session:
            return None
        out_dir = os.path.join(latest_session, "pictures")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def run(self): raise NotImplementedError

class EffectsPlanePictureGenerator(PictureGenerator):
    """Saves a static unified topological plane of all Action Effect sets."""
    def __init__(self, logs_root="logs", session_dir=None):
        super().__init__(logs_root, session_dir=session_dir)
        self.gng_colors = {}
        self.color_palette = [
            (60, 60, 220), (60, 220, 60), (220, 100, 60),
            (60, 200, 220), (200, 60, 200), (220, 200, 60),
            (100, 120, 255), (255, 120, 100), (100, 255, 120)
        ]

    def _get_color(self, name):
        if name not in self.gng_colors:
            idx = len(self.gng_colors) % len(self.color_palette)
            self.gng_colors[name] = self.color_palette[idx]
        return self.gng_colors[name]

    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        gng_dir = os.path.join(latest_session, "GNG")
        if not os.path.exists(gng_dir): return

        json_files = glob.glob(os.path.join(gng_dir, "*_effect.json"))
        if not json_files: return

        G = nx.Graph()
        node_colors = {}
        edge_colors = {}

        for json_path in json_files:
            try:
                with open(json_path, 'r') as fp:
                    data = json.load(fp)
                pred_name = os.path.splitext(os.path.basename(json_path))[0]
                color = self._get_color(pred_name)
                nodes = data.get("nodes", [])
                radiuses = data.get("local_radiuses", [])

                for i in range(len(nodes)):
                    node_id = f"{pred_name}_{i}"
                    r_val = radiuses[i] if i < len(radiuses) else 0.0
                    G.add_node(node_id, radius=r_val)
                    node_colors[node_id] = color

                edges_dict = data.get("edges", {})
                for edge_str in edges_dict.keys():
                    u_str, v_str = edge_str.split(",")
                    u_id = f"{pred_name}_{u_str}"
                    v_id = f"{pred_name}_{v_str}"
                    G.add_edge(u_id, v_id)
                    edge_colors[(u_id, v_id)] = color
                    edge_colors[(v_id, u_id)] = color
            except Exception as e:
                print(f"[EffectsPlanePictureGenerator] Error loading {json_path}: {e}")

        if len(G.nodes) == 0: return

        img = self.renderer.create_base_canvas("EFFECTS GNG PLANE", "Action Effect Sets (Eff_o)")
        self.renderer.draw_unified_GNG(img, G, node_colors, edge_colors, self.gng_colors, legend_title="Effects:")

        out_dir = self.get_pictures_dir()
        if not out_dir: return
        out_path = os.path.join(out_dir, "effects_plane.png")
        cv2.imwrite(out_path, img)
        print(f"[PictureGenerator] Saved Effects Plane to {out_path}")


class SymbolsPlanePictureGenerator(PictureGenerator):
    """Saves a static unified topological plane of discovered abstract symbols."""
    def __init__(self, logs_root="logs", session_dir=None):
        super().__init__(logs_root, session_dir=session_dir)
        self.gng_colors = {}
        self.color_palette = [
            (60, 60, 220), (60, 220, 60), (220, 100, 60),
            (60, 200, 220), (200, 60, 200), (220, 200, 60),
            (100, 120, 255), (255, 120, 100), (100, 255, 120)
        ]

    def _get_color(self, name):
        if name not in self.gng_colors:
            idx = len(self.gng_colors) % len(self.color_palette)
            self.gng_colors[name] = self.color_palette[idx]
        return self.gng_colors[name]

    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        symbols_dir = os.path.join(latest_session, "PDDL", "symbols")
        if not os.path.exists(symbols_dir): return

        json_files = glob.glob(os.path.join(symbols_dir, "*.json"))
        if not json_files: return

        G = nx.Graph()
        node_colors = {}
        edge_colors = {}

        for json_path in json_files:
            try:
                with open(json_path, 'r') as fp:
                    data = json.load(fp)
                pred_name = os.path.splitext(os.path.basename(json_path))[0]
                color = self._get_color(pred_name)
                nodes = data.get("nodes", [])
                radiuses = data.get("local_radiuses", [])

                for i in range(len(nodes)):
                    node_id = f"{pred_name}_{i}"
                    r_val = radiuses[i] if i < len(radiuses) else 0.0
                    G.add_node(node_id, radius=r_val)
                    node_colors[node_id] = color

                edges_dict = data.get("edges", {})
                for edge_str in edges_dict.keys():
                    u_str, v_str = edge_str.split(",")
                    u_id = f"{pred_name}_{u_str}"
                    v_id = f"{pred_name}_{v_str}"
                    G.add_edge(u_id, v_id)
                    edge_colors[(u_id, v_id)] = color
                    edge_colors[(v_id, u_id)] = color
            except Exception as e:
                print(f"[SymbolsPlanePictureGenerator] Error loading {json_path}: {e}")

        if len(G.nodes) == 0: return

        img = self.renderer.create_base_canvas("SYMBOLS GNG PLANE", "Grounded Abstract Preconditions")
        self.renderer.draw_unified_GNG(img, G, node_colors, edge_colors, self.gng_colors, legend_title="Symbols:")

        out_dir = self.get_pictures_dir()
        if not out_dir: return
        out_path = os.path.join(out_dir, "symbols_plane.png")
        cv2.imwrite(out_path, img)
        print(f"[PictureGenerator] Saved Symbols Plane to {out_path}")


class ActionInitPictureGenerator(PictureGenerator):
    """Saves static sensimotor prototypes for all action initiation sets (*_init.json)."""
    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        gng_dir = os.path.join(latest_session, "GNG")
        if not os.path.exists(gng_dir): return

        init_files = glob.glob(os.path.join(gng_dir, "*_init.json"))
        out_dir = self.get_pictures_dir()
        if not out_dir: return

        for json_path in init_files:
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                nodes = np.array(data.get("nodes", []))
                if len(nodes) == 0: continue

                # Compute medoid
                centroid = np.mean(nodes, axis=0)
                dists = np.linalg.norm(nodes - centroid, axis=1)
                rep_idx = int(np.argmin(dists))
                rep_node = nodes[rep_idx]

                file_base = os.path.splitext(os.path.basename(json_path))[0]
                action_name = file_base.replace("_init", "").replace("_", " ").upper()
                
                img = self.renderer.create_base_canvas(
                    f"INITIATION PRECONDITION: {action_name}",
                    f"Medoid Prototype {rep_idx} across {len(nodes)} initiation samples"
                )

                self.renderer.draw_lidar(img, rep_node[:8], is_live=True)
                self.renderer.draw_grids(img, rep_node[8:248], is_live=True)
                self.renderer.draw_sample_badge(img, n_points=len(nodes))

                out_path = os.path.join(out_dir, f"{file_base}_prototype.png")
                cv2.imwrite(out_path, img)
                print(f"[PictureGenerator] Saved action initiation image to {out_path}")
            except Exception as e:
                print(f"[ActionInitPictureGenerator] Error processing {json_path}: {e}")


class PDDLMovesPictureGenerator(PictureGenerator):
    """Parses newest domain.pddl and renders the directed possible moves graph."""
    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        domain_path = os.path.join(latest_session, "PDDL", "newest", "domain.pddl")
        if not os.path.exists(domain_path): return

        try:
            with open(domain_path, "r") as f:
                content = f.read()

            action_blocks = re.findall(r"\(:action\s+([^\s\)]+)(.*?)\s*\)\s*(?=\(:action|\)\s*\Z)", content, re.DOTALL)
            nodes = []
            edges = []

            for act_name, body in action_blocks:
                nodes.append(act_name)
                effect_match = re.search(r":effect\s+\(and(.*?)\)", body, re.DOTALL)
                if effect_match:
                    effect_str = effect_match.group(1)
                    positive_effects = re.sub(r"\(not\s+\([^\)]+\)\)", "", effect_str)
                    targets = re.findall(r"\(can_run_([^\s\)]+)\)", positive_effects)
                    for tgt in targets:
                        edges.append((act_name, tgt))

            graph_data = {"nodes": nodes, "edges": edges}
            img = self.renderer.create_base_canvas("PDDL POSSIBLE MOVES", f"Parsed from {domain_path}")
            self.renderer.draw_action_graph(img, graph_data)

            out_dir = self.get_pictures_dir()
            if not out_dir: return
            out_path = os.path.join(out_dir, "pddl_possible_moves.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved PDDL Possible Moves graph to {out_path}")
        except Exception as e:
            print(f"[PDDLMovesPictureGenerator] Error generating graph: {e}")

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
                
                # Compute statistical descriptors across all point nodes
                min_vals = np.min(nodes, axis=0)
                max_vals = np.max(nodes, axis=0)
                mean_vals = np.mean(nodes, axis=0)
                var_vals = np.var(nodes, axis=0)
                
                file_base = os.path.splitext(os.path.basename(json_path))[0]
                display_title = file_base.replace("_", " ")
                img = self.renderer.create_base_canvas(f"GNG INVARIANTS: {display_title}")
                
                # Pass min, max, and mean for all lidar directions
                lidar_data = list(zip(min_vals[:8], max_vals[:8], mean_vals[:8]))
                self.renderer.draw_lidar(img, lidar_data, is_live=False)
                
                # Pass mean and separate per-channel variance for Ganglion and Cone grids
                # Ganglion cells stats (mean, var)
                ganglion_stats = list(zip(mean_vals[8:], var_vals[8:]))

                # Compute Hue stats across sample nodes for each cone cell (48 cells total: 6x8)
                cone_hue_stats = []
                for cell_idx in range(6 * 8):
                    offset = 8 + (cell_idx * 5)
                    r_samples = nodes[:, offset + 2]
                    g_samples = nodes[:, offset + 3]
                    b_samples = nodes[:, offset + 4]
                    mean_bgr, hue_var = self.renderer.compute_hue_stats(r_samples, g_samples, b_samples)
                    cone_hue_stats.append((mean_bgr, hue_var))

                # Pass ganglion stats and hue stats into draw_grids
                self.renderer.draw_grids(img, (ganglion_stats, cone_hue_stats), is_live=False)

                # Overlay bottom-left badge showing sample point count
                self.renderer.draw_sample_badge(img, n_points=len(nodes))

                out_dir = self.get_pictures_dir()
                if not out_dir:
                    return
                out_path = os.path.join(out_dir, f"{file_base}_invariants.png")
                cv2.imwrite(out_path, img)
                print(f"[PictureGenerator] Saved GNG invariants to {out_path}")
            except Exception as e:
                print(f"[PictureGenerator] Failed to process {json_path}: {e}")

class UpdateFrequencyPictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None, timestamp_file="timestamps.txt", bin_size_sec=60.0):
        super().__init__(logs_root, session_dir=session_dir)
        self.timestamp_file = timestamp_file
        self.bin_size_sec = bin_size_sec

    def _resolve_file_path(self):
        # Checks session directory first, then root directory
        latest_session = self.get_latest_session()
        if latest_session:
            sess_path = os.path.join(latest_session, self.timestamp_file)
            if os.path.exists(sess_path):
                return sess_path
        if os.path.exists(self.timestamp_file):
            return self.timestamp_file
        return None

    def run(self):
        file_path = self._resolve_file_path()
        if not file_path:
            print(f"[PictureGenerator] File not found: {self.timestamp_file}")
            return

        try:
            with open(file_path, "r") as f:
                timestamps = [float(line.strip()) for line in f if line.strip()]
            
            img = self.renderer.create_base_canvas("UPDATE FREQUENCY OVER TIME")
            self.renderer.draw_frequency_plot(img, timestamps, bin_size_sec=self.bin_size_sec)
            self.renderer.draw_sample_badge(img, n_points=len(timestamps))

            out_dir = self.get_pictures_dir() or "."
            out_path = os.path.join(out_dir, "update_frequency.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved frequency plot to {out_path}")
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate frequency plot: {e}")


class PipelinePictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)
        self.generators = [
            EffectsPlanePictureGenerator(logs_root, session_dir=session_dir),
            SymbolsPlanePictureGenerator(logs_root, session_dir=session_dir),
            ActionInitPictureGenerator(logs_root, session_dir=session_dir),
            PDDLMovesPictureGenerator(logs_root, session_dir=session_dir),
            GNGPictureGenerator(logs_root, session_dir=session_dir),
            UpdateFrequencyPictureGenerator(logs_root, session_dir=session_dir)
        ]

    def run(self):
        for gen in self.generators:
            gen.run()