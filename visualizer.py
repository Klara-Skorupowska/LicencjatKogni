import os
import ast
import glob
import json
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, logs_root="logs"):
        self.logs_root = logs_root

    def get_latest_session(self):
        """Identifies and returns the most recent session directory."""
        if not os.path.exists(self.logs_root):
            print(f"Logs directory '{self.logs_root}' does not exist.")
            return None

        session_dirs = [
            os.path.join(self.logs_root, d)
            for d in os.listdir(self.logs_root)
            if os.path.isdir(os.path.join(self.logs_root, d))
        ]
        if not session_dirs:
            print(f"No session logs found inside '{self.logs_root}'.")
            return None

        return max(session_dirs, key=os.path.getmtime)

    def run(self):
        raise NotImplementedError


class SensimotorVisualizer(Visualizer):
    def __init__(self, logs_root="logs", color_tol=0.15, val_tol=0.12):
        super().__init__(logs_root)
        self.color_tol = color_tol
        self.val_tol = val_tol

        # Style Configuration (BGR)
        self.BG_COLOR = (248, 249, 250)
        self.BORDER_COLOR = (40, 44, 52)
        self.TEXT_MAIN = (30, 30, 30)
        self.LIDAR_RAY_BG = (225, 220, 240)
        self.LIDAR_ACTIVE = (30, 40, 180)
        self.ROBOT_BODY = (40, 44, 52)
        self.PANEL_BORDER = (210, 215, 220)
        self.IGNORE_GRAY = (128, 128, 128)

    def visualize_symbol_file(self, input_txt_path):
        MIN, MAX = [], []
        try:
            with open(input_txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MIN:"):
                        MIN = ast.literal_eval(line.replace("MIN:", "").strip())
                    elif line.startswith("MAX:"):
                        MAX = ast.literal_eval(line.replace("MAX:", "").strip())
            if not MIN or not MAX:
                return
        except Exception as e:
            print(f"Skipping {input_txt_path}: {e}")
            return

        # Canvas Setup (1000 x 620)
        img = np.ones((620, 1000, 3), dtype=np.uint8)
        img[:] = self.BG_COLOR
        cv2.rectangle(img, (0, 0), (999, 619), self.BORDER_COLOR, 3)

        # Header Card
        symbol_name = os.path.splitext(os.path.basename(input_txt_path))[0]
        category = "PRECONDITION" if "precondition" in symbol_name else "EFFECT"
        card_accent = (60, 130, 30) if category == "PRECONDITION" else (180, 80, 30)

        cv2.putText(img, symbol_name, (35, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, f"TYPE: {category}", (35, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, card_accent, 1, cv2.LINE_AA)
        cv2.line(img, (35, 85), (965, 85), self.PANEL_BORDER, 1, cv2.LINE_AA)

        # 1. Lidar Readings (first 8 values)
        center_lidar = (230, 340)
        max_radius = 160
        lidar_angles = [17, 50, 90, 150, 210, 270, 310, 343]

        cv2.putText(img, "LIDAR BOUNDS", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)

        for ang in lidar_angles:
            rad = np.deg2rad(ang - 90)
            x_end = int(center_lidar[0] + max_radius * np.cos(rad))
            y_end = int(center_lidar[1] + max_radius * np.sin(rad))
            cv2.line(img, center_lidar, (x_end, y_end), self.LIDAR_RAY_BG, 2, cv2.LINE_AA)

        if len(MIN) >= 8 and len(MAX) >= 8:
            for i, ang in enumerate(lidar_angles):
                rad = np.deg2rad(ang - 90)
                r_min = (min(MIN[i], 0.1) / 0.1) * max_radius
                r_max = (min(MAX[i], 0.1) / 0.1) * max_radius
                if r_max - r_min < 4:
                    r_max = r_min + 4

                x_min = int(center_lidar[0] + r_min * np.cos(rad))
                y_min = int(center_lidar[1] + r_min * np.sin(rad))
                x_max = int(center_lidar[0] + r_max * np.cos(rad))
                y_max = int(center_lidar[1] + r_max * np.sin(rad))
                cv2.line(img, (x_min, y_min), (x_max, y_max), self.LIDAR_ACTIVE, 4, cv2.LINE_AA)

        cv2.circle(img, center_lidar, 34, self.ROBOT_BODY, -1, cv2.LINE_AA)
        cv2.circle(img, center_lidar, 36, (100, 105, 115), 2, cv2.LINE_AA)
        cv2.line(img, (center_lidar[0], center_lidar[1] - 34), (center_lidar[0], center_lidar[1] - 18), (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Camera: 6x8 Receptive Field Grid (20x20 px each, 48 fields)
        min_cam = MIN[8:]
        max_cam = MAX[8:]
        start_x, start_y = 470, 140
        disp_scale = 3.0  # Scales each 20x20 field to 60x60 px on canvas
        rf_size = int(20 * disp_scale)
        core_size = int(10 * disp_scale)
        core_off = (rf_size - core_size) // 2

        cv2.putText(img, "RECEPTIVE FIELD GRID (6x8)", (start_x, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)

        rows, cols = 6, 8
        for r in range(rows):
            for c in range(cols):
                idx = (r * cols + c) * 5
                if idx + 4 >= len(min_cam) or idx + 4 >= len(max_cam):
                    continue

                # Parse MIN & MAX field features: [ON, OFF, R, G, B]
                on_min, off_min, r_min, g_min, b_min = min_cam[idx : idx + 5]
                on_max, off_max, r_max, g_max, b_max = max_cam[idx : idx + 5]

                # --- 1. Evaluate Color Tolerance ---
                color_dist = np.linalg.norm(np.array([r_min, g_min, b_min]) - np.array([r_max, g_max, b_max]))
                if color_dist > self.color_tol:
                    # Color difference is outside tolerance -> does not matter -> neutral gray
                    base_color = self.IGNORE_GRAY
                else:
                    avg_b = (b_min + b_max) / 2.0
                    avg_g = (g_min + g_max) / 2.0
                    avg_r = (r_min + r_max) / 2.0
                    base_color = (
                        int(np.clip(avg_b, 0.0, 1.0) * 255),
                        int(np.clip(avg_g, 0.0, 1.0) * 255),
                        int(np.clip(avg_r, 0.0, 1.0) * 255)
                    )

                # --- 2. Evaluate Value Tolerance ---
                on_val = 0.0 if abs(on_max - on_min) > self.val_tol else (on_min + on_max) / 2.0
                off_val = 0.0 if abs(off_max - off_min) > self.val_tol else (off_min + off_max) / 2.0

                # Top-left positions
                rx = start_x + (c * rf_size)
                ry = start_y + (r * rf_size)
                cx = rx + core_off
                cy = ry + core_off

                # --- 3. Render Concentric Receptive Field per Diagram ---

                # Determine center core color based on relative firing
                core_color = tuple(int(np.clip(c * on_val, 0, 255)) for c in (255, 255, 255))
                surround_color = tuple(int(np.clip(c * off_val, 0, 255)) for c in (255, 255, 255))
                
                # Draw surround
                cv2.rectangle(img, (rx, ry), (rx + rf_size, ry + rf_size), surround_color, -1)

                # Draw inner square core
                cv2.rectangle(img, (cx, cy), (cx + core_size, cy + core_size), core_color, -1)
                cv2.rectangle(img, (cx, cy), (cx + core_size, cy + core_size), self.BG_COLOR, 2)

                # Grid cell borders
                cv2.rectangle(img, (rx, ry), (rx + rf_size, ry + rf_size), self.BG_COLOR, 1)

        # Draw outer grid boundary
        cv2.rectangle(img, (start_x, start_y), (start_x + cols * rf_size, start_y + rows * rf_size), (0, 0, 0), 2)

        # Save output
        file_dir, file_name = os.path.split(input_txt_path)
        base_name = os.path.splitext(file_name)[0]
        out_dir = os.path.join(file_dir, "pictures")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base_name}_visualised.png")
        cv2.imwrite(out_path, img)
        print(f"Generated Symbol Plot: {out_path}")

    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session:
            return

        pddl_dir = os.path.join(latest_session, "PDDL", "symbols")
        if not os.path.exists(pddl_dir):
            print(f"No 'PDDL/symbols' directory located inside {latest_session}.")
            return

        target_files = glob.glob(os.path.join(pddl_dir, "*_precondition.txt")) + \
                       glob.glob(os.path.join(pddl_dir, "*_effect.txt"))

        if not target_files:
            print(f"No precondition/effect text files found in {pddl_dir}.")
            return

        for file_path in target_files:
            self.visualize_symbol_file(file_path)


class GNGVisualizer(Visualizer):
    def visualize_gng(self, json_path, output_png):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping GNG {json_path}: {e}")
            return

        G = nx.Graph()
        G.add_nodes_from(data['nodes'].keys())
        for edge in data['edges']:
            G.add_edge(str(edge['source']), str(edge['target']))

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgreen')
        nx.draw_networkx_edges(G, pos, edge_color='darkgreen', width=1.5)
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("Growing Neural Gas (State Space Network)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"Saved GNG graph image to {output_png}")

    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session:
            return

        graphs_dir = os.path.join(latest_session, "graphs", "final")
        if not os.path.exists(graphs_dir):
            print(f"No 'graphs/final' directory found inside {latest_session}.")
            return

        json_files = glob.glob(os.path.join(graphs_dir, "*.json"))
        if not json_files:
            print(f"No GNG JSON graphs found inside {graphs_dir}.")
            return

        out_dir = os.path.join(graphs_dir, "pictures")
        os.makedirs(out_dir, exist_ok=True)

        for json_path in json_files:
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            out_png = os.path.join(out_dir, f"{base_name}_visualised.png")
            self.visualize_gng(json_path, out_png)

class TransitionalMapVisualizer(Visualizer):
    def visualize_transitional_map(self, graphml_path, output_png):
        try:
            # GraphML uses a directed graph for state transitions
            G = nx.read_graphml(graphml_path)
            if not isinstance(G, nx.DiGraph):
                G = nx.DiGraph(G)
        except Exception as e:
            print(f"Skipping GraphML {graphml_path}: {e}")
            return

        fig, ax = plt.subplots(figsize=(13, 11))
        pos = nx.spring_layout(G, seed=42, k=1.8)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, node_color='lightskyblue', edgecolors='midnightblue', linewidths=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

        # Separate bidirectional edges from unidirectional ones
        rad = 0.25
        for u, v, data in G.edges(data=True):
            skill = data.get('skill', '')
            weight = data.get('weight', '')
            label = f"{skill} ({weight})" if skill and weight != '' else f"{skill}{weight}"

            is_bidirectional = G.has_edge(v, u)
            curve = rad if is_bidirectional else 0.0

            # Draw curved directed edge
            arrow = ax.annotate(
                "",
                xy=pos[v],
                xytext=pos[u],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="steelblue",
                    lw=1.8,
                    shrinkA=18,
                    shrinkB=18,
                    connectionstyle=f"arc3,rad={curve}"
                )
            )

            # Compute label position shifted along the curve
            p_start = np.array(pos[u])
            p_end = np.array(pos[v])
            mid = 0.5 * (p_start + p_end)
            diff = p_end - p_start
            length = np.linalg.norm(diff)

            if length > 0:
                # Perpendicular unit normal vector
                normal = np.array([-diff[1], diff[0]]) / length
                # Offset position along the curve toward the arc direction
                text_pos = mid + normal * (curve * 0.45 * length)
            else:
                text_pos = mid

            # Draw readable label box
            ax.text(
                text_pos[0], text_pos[1],
                label,
                size=8,
                weight='semibold',
                ha="center",
                va="center",
                color="darkblue",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="steelblue", lw=0.8, alpha=0.92),
                zorder=5
            )

        ax.set_title("Transitional Map (Action-State Graph)", fontsize=13, weight='bold')
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"Saved Transitional Map image to {output_png}")

    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session:
            return

        graphs_dir = os.path.join(latest_session, "graphs", "final")
        if not os.path.exists(graphs_dir):
            print(f"No 'graphs/final' directory found inside {latest_session}.")
            return

        graphml_files = glob.glob(os.path.join(graphs_dir, "*.graphml"))
        if not graphml_files:
            print(f"No GraphML files found inside {graphs_dir}.")
            return

        out_dir = os.path.join(graphs_dir, "pictures")
        os.makedirs(out_dir, exist_ok=True)

        for graphml_path in graphml_files:
            base_name = os.path.splitext(os.path.basename(graphml_path))[0]
            out_png = os.path.join(out_dir, f"{base_name}_visualised.png")
            self.visualize_transitional_map(graphml_path, out_png)


class PipelineVisualizer:
    def __init__(self, logs_root="logs"):
        self.visualizers = [
            SensimotorVisualizer(logs_root),
            GNGVisualizer(logs_root),
            TransitionalMapVisualizer(logs_root),
        ]

    def run_all(self):
        for visualizer in self.visualizers:
            visualizer.run()


if __name__ == "__main__":
    runner = PipelineVisualizer(logs_root="logs")
    runner.run_all()