import os
import ast
import glob
import json
import time
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from communicator import Communicator

# =============================================================================
# SHARED RENDERERS & UTILS
# =============================================================================

class SensimotorRenderer:
    """Shared drawing logic for both static and live sensimotor visualizations."""
    
    # Style Configuration (BGR)
    BG_COLOR = (248, 249, 250)
    BORDER_COLOR = (40, 44, 52)
    TEXT_MAIN = (30, 30, 30)
    TEXT_MUTED = (100, 105, 115)
    LIDAR_RAY_BG = (225, 220, 240)  # Pink projection lines
    LIDAR_ACTIVE = (30, 40, 180)    # Red active lines
    ROBOT_BODY = (40, 44, 52)
    PANEL_BORDER = (210, 215, 220)
    
    # Grid Colors
    GRID_BLACK = (0, 0, 0)
    GRID_DARK_GRAY = (80, 80, 80)
    GRID_NEUTRAL = (200, 200, 200)

    LIDAR_ANGLES = [17, 50, 90, 150, 210, 270, 310, 343]

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

        # Draw background pink rays
        for ang in self.LIDAR_ANGLES:
            rad = np.deg2rad(ang - 90)
            x_end = int(center[0] + max_radius * np.cos(rad))
            y_end = int(center[1] + max_radius * np.sin(rad))
            cv2.line(img, center, (x_end, y_end), self.LIDAR_RAY_BG, 2, cv2.LINE_AA)

        # Draw active readings
        for i, ang in enumerate(self.LIDAR_ANGLES):
            rad = np.deg2rad(ang - 90)
            
            if is_live:
                val = max(min(float(lidar_data[i]), 0.1), 0.0)
                r_val = (val / 0.1) * max_radius
                
                x_val = int(center[0] + r_val * np.cos(rad))
                y_val = int(center[1] + r_val * np.sin(rad))
                cv2.line(img, center, (x_val, y_val), self.LIDAR_ACTIVE, 3, cv2.LINE_AA)
                cv2.circle(img, (x_val, y_val), 4, self.LIDAR_ACTIVE, -1, cv2.LINE_AA)
            else:
                # Static data is a tuple of (min, max)
                r_min = (min(lidar_data[i][0], 0.1) / 0.1) * max_radius
                r_max = (min(lidar_data[i][1], 0.1) / 0.1) * max_radius
                if r_max - r_min < 4:
                    r_max = r_min + 4
                    
                x_min = int(center[0] + r_min * np.cos(rad))
                y_min = int(center[1] + r_min * np.sin(rad))
                x_max = int(center[0] + r_max * np.cos(rad))
                y_max = int(center[1] + r_max * np.sin(rad))
                cv2.line(img, (x_min, y_min), (x_max, y_max), self.LIDAR_ACTIVE, 4, cv2.LINE_AA)
                cv2.circle(img, (x_max, y_max), 4, self.LIDAR_ACTIVE, -1, cv2.LINE_AA)

        # Robot body
        cv2.circle(img, center, 34, self.ROBOT_BODY, -1, cv2.LINE_AA)
        cv2.circle(img, center, 36, (100, 105, 115), 2, cv2.LINE_AA)
        cv2.line(img, (center[0], center[1] - 34), (center[0], center[1] - 18), (255, 255, 255), 2, cv2.LINE_AA)

    def draw_wheels(self, img, left, right, max_vel, center=(230, 520)):
        cv2.putText(img, "Wheels", (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "L", (165, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "R", (265, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        
        # Draw Zero Line
        cv2.line(img, (140, center[1]), (290, center[1]), (200, 200, 200), 2, cv2.LINE_AA)

        max_h = 40  # Max height in one direction
        max_vel = max(max_vel, 0.001)  # Prevent division by zero

        l_h = int(np.clip((left / max_vel) * max_h, -max_h, max_h))
        r_h = int(np.clip((right / max_vel) * max_h, -max_h, max_h))

        # Left bar (Draws up for positive, down for negative)
        cv2.rectangle(img, (150, center[1] - l_h), (190, center[1]), (0, 0, 0), -1)
        # Right bar
        cv2.rectangle(img, (250, center[1] - r_h), (290, center[1]), (0, 0, 0), -1)

    def draw_grids(self, img, camera_data, is_live=True):
        rows, cols = 120//16, 160//16
        cell_size = 210//rows
        
        # Coordinates for the two grids
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
                else:
                    # Properly unpack the 5 tuples generated by zip(MIN, MAX) for this cell
                    (on_min, on_max), (off_min, off_max), (r_min, r_max), (g_min, g_max), (b_min, b_max) = camera_data[idx: idx + 5]
                    
                    # Averages the min/max and evaluates to True if > 0.5
                    on_bool = (on_min + on_max) / 2.0 > 0.5
                    off_bool = (off_min + off_max) / 2.0 > 0.5
                    red = (r_min + r_max) / 2.0
                    green = (g_min + g_max) / 2.0
                    blue = (b_min + b_max) / 2.0

                # 1. Receptive Field Logic (On/Off)
                if on_bool and not off_bool:
                    rf_color = self.GRID_BLACK
                elif off_bool and not on_bool:
                    rf_color = self.GRID_DARK_GRAY
                else:
                    rf_color = self.GRID_NEUTRAL

                rf_x = rf_start_x + (c * cell_size)
                rf_y = rf_start_y + (r * cell_size)
                cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), rf_color, -1)

                # 2. Color Perception Logic (RGB)
                if max(red, green, blue) <= 1.0:
                    red, green, blue = red * 255.0, green * 255.0, blue * 255.0
                cp_color = (int(np.clip(blue, 0, 255)), int(np.clip(green, 0, 255)), int(np.clip(red, 0, 255)))
                
                cp_x = c_start_x + (c * cell_size)
                cp_y = c_start_y + (r * cell_size)
                cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), cp_color, -1)


def render_transitional_figure(G):
    """Shared method for drawing Transitional Maps."""
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    if len(G.nodes) == 1:
        pos = {list(G.nodes)[0]: (0.0, 0.0)}
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    else:
        pos = nx.spring_layout(G, seed=42, k=2.5, iterations=150)

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    unique_skills = sorted(list({data.get('skill', 'unknown') for _, _, data in G.edges(data=True)}))
    skill_colors = {skill: palette[i % len(palette)] for i, skill in enumerate(unique_skills)}

    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=2000, node_color='#eef2f7',
        edgecolors='#2c3e50', linewidths=2.0
    )
    if nodes is not None: nodes.set_zorder(5)
    
    labels = nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold", font_color="#1a1a1a")
    for text_obj in labels.values(): text_obj.set_zorder(7)

    for node, data in G.nodes(data=True):
        raw_aliases = data.get('aliases')
        if not raw_aliases: continue

        if isinstance(raw_aliases, str):
            try: aliases_list = ast.literal_eval(raw_aliases)
            except (ValueError, SyntaxError): aliases_list = [raw_aliases]
        else: aliases_list = list(raw_aliases)

        alias_str = ", ".join(str(a) for a in aliases_list)[:25] + ("..." if len(", ".join(str(a) for a in aliases_list)) > 28 else "")
        x, y = pos[node]
        ax.text(x, y - 0.16, f"aliases: {alias_str}", fontsize=8, weight='medium', ha="center", va="top",
                color="#2c3e50", bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#b0bec5", lw=0.9, alpha=0.95), zorder=8)

    edge_groups = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_groups.setdefault((u, v), []).append((key, data))

    mutation_scale = 14
    base_angles = [90, 30, 150, 210, 330, 270]

    for (u, v), edges in edge_groups.items():
        num_edges = len(edges)
        has_reverse = G.has_edge(v, u)

        for i, (key, data) in enumerate(edges):
            skill = data.get('skill', 'unknown')
            edge_color = skill_colors.get(skill, "#1f77b4")

            if u == v:
                center_deg = base_angles[i % len(base_angles)]
                
                # Increase base arm length from 48 to 100, and the step size for multiple loops to 30
                arm = 100 + 30 * (i // len(base_angles))
                
                # Increase rad from 12 to 25 for a wider curve
                loop = FancyArrowPatch(
                    pos[u], pos[u], 
                    arrowstyle="-|>",
                    connectionstyle=f"arc,angleA={center_deg-28},angleB={center_deg+28},armA={arm},armB={arm},rad=25",
                    mutation_scale=mutation_scale, 
                    color=edge_color, 
                    lw=2.0, 
                    shrinkA=25,
                    shrinkB=25, 
                    zorder=4
                )
                ax.add_patch(loop)
                continue
            curve = (0.25 + ((i - (num_edges - 1) / 2.0) * 0.15 if num_edges > 1 else 0.0)) if has_reverse else ((i - (num_edges - 1) / 2.0) * 0.20 if num_edges > 1 else 0.0)
            
            arrow = FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>", connectionstyle=f"arc3,rad={curve}",
                                    mutation_scale=mutation_scale, color=edge_color, lw=2.0, shrinkA=20, shrinkB=20, zorder=4)
            ax.add_patch(arrow)

    ax.set_title("Transitional Map (Action-State Graph)", fontsize=13, weight='bold', pad=25)
    legend_handles = [mpatches.Patch(facecolor=color, edgecolor="#333333", lw=0.5, label=skill) for skill, color in skill_colors.items()]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(5, len(legend_handles)),
                  frameon=True, facecolor="#ffffff", edgecolor="#bdc3c7", fontsize=8)

    ax.margins(0.20)
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_gng_figure(data):
    """Shared method for drawing GNG Maps."""
    G = nx.Graph()
    G.add_nodes_from(data['nodes'].keys())
    for edge in data['edges']:
        G.add_edge(str(edge['source']), str(edge['target']))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='lightgreen')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='darkgreen', width=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    
    ax.set_title("Growing Neural Gas (State Space Network)")
    ax.axis("off")
    fig.tight_layout()
    return fig

# =============================================================================
# STATIC VISUALIZERS
# =============================================================================

class Visualizer:
    def __init__(self, logs_root="logs"):
        self.logs_root = logs_root

    def get_latest_session(self):
        if not os.path.exists(self.logs_root): return None
        session_dirs = [os.path.join(self.logs_root, d) for d in os.listdir(self.logs_root) if os.path.isdir(os.path.join(self.logs_root, d))]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def run(self): raise NotImplementedError


class SensimotorVisualizer(Visualizer, SensimotorRenderer):
    def visualize_symbol_file(self, input_txt_path):
        MIN, MAX = [], []
        predicate_label, radius_val, aliases_val = None, None, None

        try:
            with open(input_txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("PREDICATE:"): predicate_label = line.replace("PREDICATE:", "").strip()
                    elif line.startswith("RADIUS:"): radius_val = line.replace("RADIUS:", "").strip()
                    elif line.startswith("ALIASES:"): aliases_val = line.replace("ALIASES:", "").strip()
                    elif line.startswith("MIN:"): MIN = ast.literal_eval(line.replace("MIN:", "").strip())
                    elif line.startswith("MAX:"): MAX = ast.literal_eval(line.replace("MAX:", "").strip())
            if not MIN or not MAX: return
        except Exception as e:
            print(f"[Visualizer] Skipping {input_txt_path}: {e}")
            return

        file_base = os.path.splitext(os.path.basename(input_txt_path))[0]
        display_name = predicate_label if predicate_label else f"(at {file_base})"
        sub_header = f"GROUNDED OBJECT: {file_base}"
        if radius_val: sub_header += f"  |  RADIUS: {radius_val}"
        if aliases_val: sub_header += f"  |  ALIASES: {aliases_val}"

        img = self.create_base_canvas(display_name, sub_header)

        # Re-format MIN/MAX data so shared renderer can process it
        lidar_data = list(zip(MIN[:8], MAX[:8]))
        self.draw_lidar(img, lidar_data, is_live=False)

        camera_data = list(zip(MIN[8:], MAX[8:]))
        self.draw_grids(img, camera_data, is_live=False)

        # Output Generation
        out_dir = os.path.join(os.path.split(input_txt_path)[0], "pictures")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{file_base}_visualised.png")
        cv2.imwrite(out_path, img)
        print(f"[Visualizer] Generated Symbol Plot: {out_path}")

    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        symbols_dir = os.path.join(latest_session, "PDDL", "symbols")
        if not os.path.exists(symbols_dir): return
        for file_path in glob.glob(os.path.join(symbols_dir, "*.txt")):
            self.visualize_symbol_file(file_path)


class GNGVisualizer(Visualizer):
    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        graphs_dir = os.path.join(latest_session, "graphs")
        
        for json_path in glob.glob(os.path.join(graphs_dir, "*.json")):
            with open(json_path, 'r') as f:
                fig = render_gng_figure(json.load(f))
            
            out_dir = os.path.join(graphs_dir, "pictures")
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(json_path))[0]}_visualised.png")
            
            fig.savefig(out_png, dpi=300)
            plt.close(fig)
            print(f"[Visualizer] Saved GNG graph image to {out_png}")


class TransitionalMapVisualizer(Visualizer):
    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        graphs_dir = os.path.join(latest_session, "graphs")

        for graphml_path in glob.glob(os.path.join(graphs_dir, "*.graphml")):
            G = nx.read_graphml(graphml_path)
            fig = render_transitional_figure(G)

            out_dir = os.path.join(graphs_dir, "pictures")
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(graphml_path))[0]}_visualised.png")
            
            fig.savefig(out_png, dpi=300)
            plt.close(fig)
            print(f"[Visualizer] Saved Transitional Map image to {out_png}")


class PipelineVisualizer:
    def __init__(self, logs_root="logs"):
        self.visualizers = [SensimotorVisualizer(logs_root), GNGVisualizer(logs_root), TransitionalMapVisualizer(logs_root)]

    def run_all(self):
        for visualizer in self.visualizers:
            visualizer.run()

# =============================================================================
# LIVE MONITORING 
# =============================================================================

class LiveSensimotorMonitor(SensimotorRenderer):
    def __init__(self, bus: Communicator, timeout: float = 0.1):
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

        # 1. Fetch LiDAR data
        lidar_readings = []
        for direction in self.LIDAR_ANGLES:
            val = self._safe_call(f"/lidar/{direction}/ask/value")
            if val is None: return
            lidar_readings.append(val)

        # 2. Fetch Camera data
        raw_coded = self._safe_call("/camera/ask/coded")
        if raw_coded is None: return
        cam_array = np.asarray(raw_coded, dtype=float).flatten()
        if len(cam_array) < (6 * 8 * 5): return

        # 3. Fetch Wheels data
        speeds = self._safe_call("/wheels/ask/speed")
        max_vel = self._safe_call("/get/wheels/max_velocity")
        if speeds is None or max_vel is None: return

        # Render
        img = self.create_base_canvas("LIVE SENSIMOTOR MONITOR")
        self.draw_lidar(img, lidar_readings, is_live=True)
        self.draw_wheels(img, speeds[0], speeds[1], float(max_vel))
        self.draw_grids(img, cam_array, is_live=True)

        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


class LiveGraphMonitor(Visualizer):
    def __init__(self, logs_root="logs", gng_filename="gng_graph.json", trans_filename="transitional_graph.graphml", poll_interval=1.0):
        super().__init__(logs_root)
        self.gng_filename = gng_filename
        self.trans_filename = trans_filename
        self.poll_interval = poll_interval

        self._last_poll_time = 0.0
        self._last_gng_mtime = 0.0
        self._last_trans_mtime = 0.0

        self.gng_window = "Live GNG Monitor"
        self.trans_window = "Live Transitional Map Monitor"
        self._windows_created = False

    def _ensure_windows(self):
        if not self._windows_created:
            cv2.namedWindow(self.gng_window, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(self.trans_window, cv2.WINDOW_AUTOSIZE)

            placeholder = np.full((300, 400, 3), 40, dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for data...", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow(self.gng_window, placeholder)
            cv2.imshow(self.trans_window, placeholder)
            self._windows_created = True

    def _fig_to_cv2(self, fig):
        fig.canvas.draw()
        bgr_img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return bgr_img

    def update(self):
        self._ensure_windows()
        cv2.waitKey(1)

        now = time.time()
        if now - self._last_poll_time < self.poll_interval: return
        self._last_poll_time = now

        latest_session = self.get_latest_session()
        if not latest_session: return

        live_dir = os.path.join(latest_session, "graphs")
        gng_path = os.path.join(live_dir, self.gng_filename)
        trans_path = os.path.join(live_dir, self.trans_filename)

        try:
            if os.path.exists(gng_path) and os.path.getsize(gng_path) > 0:
                mtime = os.path.getmtime(gng_path)
                if mtime > self._last_gng_mtime:
                    with open(gng_path, "r") as f:
                        cv2.imshow(self.gng_window, self._fig_to_cv2(render_gng_figure(json.load(f))))
                    self._last_gng_mtime = mtime
        except Exception: pass

        try:
            if os.path.exists(trans_path) and os.path.getsize(trans_path) > 0:
                mtime = os.path.getmtime(trans_path)
                if mtime > self._last_trans_mtime:
                    cv2.imshow(self.trans_window, self._fig_to_cv2(render_transitional_figure(nx.read_graphml(trans_path))))
                    self._last_trans_mtime = mtime
        except Exception: pass

    def close(self):
        if self._windows_created:
            cv2.destroyWindow(self.gng_window)
            cv2.destroyWindow(self.trans_window)
            self._windows_created = False

if __name__ == "__main__":
    runner = PipelineVisualizer(logs_root="logs")
    runner.run_all()