import os
import re
import glob
import json
import time
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
        # Fresh, professional blue palette (BGR format for OpenCV)
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
        """
        Converts RGB vectors to Hue (0 to 2pi) and computes circular mean and variance.
        Returns:
            mean_rgb_tuple: (B, G, R) color corresponding to the mean hue (full saturation & value)
            circ_var: Circular variance in range [0, 1]
        """
        # Ensure values are in float [0, 1]
        max_v = max(np.max(r_vals), np.max(g_vals), np.max(b_vals))
        if max_v > 1.0:
            r = np.array(r_vals, dtype=np.float32) / 255.0
            g = np.array(g_vals, dtype=np.float32) / 255.0
            b = np.array(b_vals, dtype=np.float32) / 255.0
        else:
            r = np.array(r_vals, dtype=np.float32)
            g = np.array(g_vals, dtype=np.float32)
            b = np.array(b_vals, dtype=np.float32)

        # Standard RGB to Hue conversion
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        h = np.zeros_like(r)
        nonzero = delta > 1e-6

        # Red is max
        mask = nonzero & (cmax == r)
        h[mask] = (60.0 * (((g[mask] - b[mask]) / delta[mask]) % 6))

        # Green is max
        mask = nonzero & (cmax == g)
        h[mask] = (60.0 * (((b[mask] - r[mask]) / delta[mask]) + 2))

        # Blue is max
        mask = nonzero & (cmax == b)
        h[mask] = (60.0 * (((r[mask] - g[mask]) / delta[mask]) + 4))

        # Convert degrees [0, 360) to radians [0, 2pi)
        angles = np.deg2rad(h)

        # Directional statistics: mean resultant vector
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        R = np.hypot(sin_mean, cos_mean)
        circ_var = 1.0 - R  # 0: perfectly invariant, 1: completely dispersed

        mean_angle_deg = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0

        # Convert mean hue back to pure BGR color (OpenCV H: 0-179, S: 255, V: 255)
        hsv_pixel = np.uint8([[[int(mean_angle_deg / 2.0), 255, 255]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        mean_bgr = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

        return mean_bgr, circ_var

    def _rgb_to_hue_bgr(self, r, g, b):
        """
        Converts RGB values [0.0, 1.0] to a pure Hue color in BGR format.
        Preserves grayscale/achromatic pixels if saturation is near zero.
        """
        # Clamp inputs
        r_c = np.clip(float(r), 0.0, 1.0)
        g_c = np.clip(float(g), 0.0, 1.0)
        b_c = np.clip(float(b), 0.0, 1.0)

        # Handle grayscale/achromatic pixels where Hue is undefined
        max_c = max(r_c, g_c, b_c)
        min_c = min(r_c, g_c, b_c)
        delta = max_c - min_c

        # If low saturation (e.g. gray, white, dark road), preserve the neutral luminance
        if max_c < 0.05 or (delta / max_c) < 0.15:
            val = int(max_c * 255)
            return (val, val, val)

        # Convert uint8 RGB -> HSV -> BGR so OpenCV handles ranges consistently
        rgb_u8 = np.array([[[int(r_c * 255), int(g_c * 255), int(b_c * 255)]]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        h = hsv[0, 0, 0]  # OpenCV uint8 Hue is naturally in [0, 179]

        # Full saturation and brightness for pure Hue representation
        pure_hsv = np.array([[[h, 255, 255]]], dtype=np.uint8)
        pure_bgr = cv2.cvtColor(pure_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(pure_bgr[0]), int(pure_bgr[1]), int(pure_bgr[2]))
    
    def draw_sample_badge(self, img, n_points, position=(830, 48)):
        """Draws a badge indicating the number of sample points used."""
        text = f"N = {n_points} samples"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = position
        pad_x, pad_y = 10, 6
        
        # Badge background pill
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

        # Draw background radial guides
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
                # Expects tuple: (val_min, val_max, val_mean)
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
                
                # Range bar covering min to max values
                cv2.line(img, (x_min, y_min), (x_max, y_max), self.secondary_color, 4, cv2.LINE_AA)
                # Dot for the mean reading along the ray
                cv2.circle(img, (x_mean, y_mean), 5, self.primary_color, -1, cv2.LINE_AA)

        # Robot chassis
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
        cv2.rectangle(img, (150, center[1] - max_h), (190, center[1] + max_h), self.tertiary_color, -1)
        cv2.rectangle(img, (250, center[1] - max_h), (290, center[1] + max_h), self.tertiary_color, -1)

        # Active velocities in primary color
        cv2.rectangle(img, (150, center[1] - l_h), (190, center[1]), self.primary_color, -1)
        cv2.rectangle(img, (250, center[1] - r_h), (290, center[1]), self.primary_color, -1)

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

                    # 1. Ganglion cell (check ON/OFF invariance)
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

                    # 2. Cone cells: Hue circular variance checking and mean hue rendering
                    mean_bgr, hue_circ_var = cone_hue_stats[cell_num]

                    if hue_circ_var <= self.INVARIANT_THRESHOLD:
                        # Fill with the reconstructed mean hue color
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), mean_bgr, -1)
                    else:
                        # Hollow border if the hue varies significantly
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), self.PANEL_BORDER, 1)

    def draw_transition_graph(self, img, graph_data):
        """Draws the transition graph directly onto the OpenCV image canvas."""
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            cv2.putText(img, "No graph data", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, self.neutral_gray, 2)
            return

        center = (500, 350)
        node_radius = 50
        
        # 1. Build a NetworkX Directed Graph specifically for the layout algorithm
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for edge_info in edges:
            source = edge_info.get("source")
            target = edge_info.get("target")
            count = edge_info.get("count", 1)
            
            # Using count as weight naturally pulls frequently transitioning nodes closer
            if source in nodes and target in nodes:
                G.add_edge(source, target, weight=count)
                
        # 2. Compute Force-Directed / Spring Layout
        # 'k' controls the optimal distance between nodes. Increasing it forces more spacing.
        optimal_dist = node_radius / math.sqrt(max(len(nodes), 1)) 
        raw_pos = nx.spring_layout(G, k=optimal_dist, center=center, scale=210, seed=42)
        
        # 3. Convert float coordinates to integer pixels for OpenCV
        positions = {node: (int(coords[0]), int(coords[1])) for node, coords in raw_pos.items()}

        # Dynamic thickness scaling limits
        thickness = 2
        counts = [edge.get("count", 1) for edge in edges]
        max_count = max(counts) if counts else 1
        min_count = min(counts) if counts else 1

        # Draw Edges
        for edge_info in edges:
            source = edge_info.get("source")
            target = edge_info.get("target")
            count = edge_info.get("count", 1)

            if source not in positions or target not in positions:
                continue

            pt1 = positions[source]
            pt2 = positions[target]
                    
            if max_count == min_count:
                t = 1.0
            else:
                t = (count - min_count) / (max_count - min_count)
            faint_color = tuple(int(0.85 * bg + 0.15 * pri) for bg, pri in zip(self.BG_COLOR, self.primary_color))
            edge_color = tuple(int((1.0 - t) * faint_color[c] + t * self.primary_color[c]) for c in range(3))

            # Math to draw arrow exactly to the edge of the circle, not center
            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            dist = math.hypot(dx, dy)

            if dist == 0:
                continue

            start_x = int(pt1[0] + (node_radius * dx / dist))
            start_y = int(pt1[1] + (node_radius * dy / dist))
            end_x = int(pt2[0] - (node_radius * dx / dist))
            end_y = int(pt2[1] - (node_radius * dy / dist))

            # Set up points for the Bezier curve
            P1 = np.array([start_x, start_y], dtype=float)
            P2 = np.array([end_x, end_y], dtype=float)
            
            gap_dx = P2[0] - P1[0]
            gap_dy = P2[1] - P1[1]
            gap_dist = math.hypot(gap_dx, gap_dy)
            
            if gap_dist > 0:
                M = (P1 + P2) / 2.0
                
                # Normal vector perpendicular to the edge direction
                nx = -gap_dy / gap_dist
                ny = gap_dx / gap_dist
                
                # Push the control point out dynamically (15% of the edge length)
                curve_offset = gap_dist * 0.2  
                C = M + np.array([nx, ny]) * curve_offset
                
                # Generate 20 points along the quadratic Bezier curve
                t = np.linspace(0, 1, 20).reshape(-1, 1)
                curve_pts = ((1 - t)**2 * P1 + 2 * (1 - t) * t * C + t**2 * P2).astype(np.int32)
                
                # Draw the curved line
                cv2.polylines(
                    img, 
                    [curve_pts], 
                    isClosed=False, 
                    color=edge_color, 
                    thickness=thickness, 
                    lineType=cv2.LINE_AA
                )
                
                # Calculate the tangent at the end of the curve for the arrowhead
                tx = P2[0] - C[0]
                ty = P2[1] - C[1]
                t_len = math.hypot(tx, ty)
                
                if t_len > 0:
                    tx /= t_len
                    ty /= t_len
                    
                    # Create an artificial starting point 20 pixels back to draw a consistent arrow
                    arrow_start = (int(P2[0] - tx * 20), int(P2[1] - ty * 20))
                    arrow_end = (int(P2[0]), int(P2[1]))
                    
                    cv2.arrowedLine(
                        img,
                        arrow_start,
                        arrow_end,
                        edge_color,
                        thickness,
                        tipLength=0.4,
                        line_type=cv2.LINE_AA
                    )
                    

        # Draw Nodes (drawn after edges so edges don't overlap node circles)
        for node, (x, y) in positions.items():
            cv2.circle(img, (x, y), node_radius, self.tertiary_color, -1)
            cv2.circle(img, (x, y), node_radius, self.primary_color, 2)
                
            # Replace underscores and wrap text to fit inside the circle width
            display_name = node.replace('_', ' ')
            wrapped_lines = textwrap.wrap(display_name, width=10)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            
            # Measure all lines to calculate total text block height
            line_metrics = []
            for line in wrapped_lines:
                (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                line_metrics.append((line, tw, th, baseline))
                
            total_height = sum(th + baseline for _, _, th, baseline in line_metrics) + max(0, len(wrapped_lines) - 1) * 3
            
            # Center the multi-line text block vertically and horizontally inside the circle
            current_y = y - total_height // 2
            
            for line, tw, th, baseline in line_metrics:
                txt_x = int(x - tw / 2)
                txt_y = int(current_y + th)
                cv2.putText(img, line, (txt_x, txt_y), font, font_scale, self.TEXT_MAIN, thickness, cv2.LINE_AA)
                current_y += th + baseline + 3

    def draw_unified_GNG(self, img, G, node_colors, edge_colors, gng_colors):
        """Draws the unified topological GNG plane and legend onto the canvas."""
        if len(G.nodes) == 0:
            return

        # Calculate 2D layout projection
        pos = nx.spring_layout(G, seed=42, k=0.25)

        # 1. Determine raw bounding box of the layout
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        raw_w = max_x - min_x if max_x > min_x else 1.0
        raw_h = max_y - min_y if max_y > min_y else 1.0

        # 2. Determine maximum radius size in pixels to prevent clipping
        max_r_px = max([max(int(G.nodes[n].get("radius", 0.0) * 35), 5) for n in G.nodes()], default=5)

        # 3. Calculate dynamic scale and center offsets
        pad_left = max_r_px + 30
        pad_right = max_r_px + 200  # Extra padding on the right to avoid the legend
        pad_y_top = max_r_px + 100  # Extra padding on top to avoid the title
        pad_y_bot = max_r_px + 30
        
        avail_w = 1000 - (pad_left + pad_right)
        avail_h = 620 - (pad_y_top + pad_y_bot)
        
        scale = min(avail_w / raw_w, avail_h / raw_h)
        
        center_x = pad_left + avail_w / 2 - ((min_x + max_x) / 2) * scale
        center_y = pad_y_top + avail_h / 2 - ((min_y + max_y) / 2) * scale

        # 1. Draw local radiuses (Covered Area)
        for node in G.nodes():
            pt = (int(center_x + pos[node][0] * scale), int(center_y + pos[node][1] * scale))
            base_color = node_colors.get(node, self.neutral_gray)
            
            # Blend 75% background with 25% predicate color for a lighter shade
            light_color = tuple(int(0.75 * bg + 0.25 * bc) for bg, bc in zip(self.BG_COLOR, base_color))
            
            # Fetch radius, default to 0 if missing, and scale it for the screen
            r_val = G.nodes[node].get("radius", 0.0)
            if r_val > 0:
                # 35 is a visual scaling multiplier to make the radius readable on the 1000x620 canvas
                r_px = max(int(r_val * 35), 5) 
                cv2.circle(img, pt, r_px, light_color, -1, cv2.LINE_AA)

        # 2. Draw edges
        for u, v in G.edges():
            pt1 = (int(center_x + pos[u][0] * scale), int(center_y + pos[u][1] * scale))
            pt2 = (int(center_x + pos[v][0] * scale), int(center_y + pos[v][1] * scale))
            color = edge_colors.get((u, v), self.neutral_gray)
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)

        # 3. Draw nodes
        for node in G.nodes():
            pt = (int(center_x + pos[node][0] * scale), int(center_y + pos[node][1] * scale))
            color = node_colors.get(node, self.neutral_gray)
            cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, self.TEXT_MAIN, 1, cv2.LINE_AA)

        # 4. Draw legend
        legend_start_y = 120
        cv2.putText(img, "Predicates:", (720, legend_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for i, (pred_name, color) in enumerate(gng_colors.items()):
            y_pos = legend_start_y + 25 + (i * 20)
            cv2.circle(img, (730, y_pos - 4), 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, pred_name, (745, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_MAIN, 1, cv2.LINE_AA)


    def draw_frequency_plot(self, img, timestamps, bin_size_sec=60.0):
        """Draws an update frequency histogram / rate plot over time."""
        if not timestamps or len(timestamps) < 2:
            cv2.putText(img, "Insufficient timestamp data", (350, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neutral_gray, 2, cv2.LINE_AA)
            return

        ts = np.sort(np.array(timestamps, dtype=float))
        t_start, t_end = ts[0], ts[-1]
        duration = max(t_end - t_start, 1.0)
        
        # Bin timestamps to calculate frequency (events per bin)
        n_bins = max(int(np.ceil(duration / bin_size_sec)), 1)
        bin_edges = np.linspace(t_start, t_start + n_bins * bin_size_sec, n_bins + 1)
        counts, _ = np.histogram(ts, bins=bin_edges)

        # Plot bounding box dimensions
        plot_x, plot_y = 90, 140
        plot_w, plot_h = 820, 380
        cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.PANEL_BORDER, 1)

        # Summary Metrics
        avg_freq = len(ts) / (duration / 60.0)  # updates per minute
        stat_text = f"Total: {len(ts)} updates | Duration: {duration:.1f}s | Avg Rate: {avg_freq:.2f} updates/min"
        cv2.putText(img, stat_text, (plot_x, plot_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)

        max_count = max(int(np.max(counts)), 1)
        
        # Draw horizontal gridlines & Y-axis labels
        grid_steps = 4
        for i in range(grid_steps + 1):
            y_val = plot_y + plot_h - int(i * (plot_h / grid_steps))
            val_label = f"{int(i * (max_count / grid_steps))}"
            cv2.line(img, (plot_x, y_val), (plot_x + plot_w, y_val), self.tertiary_color, 1, cv2.LINE_AA)
            cv2.putText(img, val_label, (plot_x - 35, y_val + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.neutral_gray, 1, cv2.LINE_AA)

        # Draw histogram bars and update frequency trend
        bar_w = max(int(plot_w / n_bins) - 3, 2)
        curve_pts = []

        for i, count in enumerate(counts):
            bar_h = int((count / max_count) * (plot_h - 20))
            bx = plot_x + int(i * (plot_w / n_bins)) + 2
            by = plot_y + plot_h - bar_h

            # Bar representing update count in that time interval
            cv2.rectangle(img, (bx, by), (bx + bar_w, plot_y + plot_h), self.secondary_color, -1)
            cv2.rectangle(img, (bx, by), (bx + bar_w, plot_y + plot_h), self.primary_color, 1)
            
            # Point for interpolation line
            pt_center = (bx + bar_w // 2, by)
            curve_pts.append(pt_center)

        if len(curve_pts) > 1:
            for i in range(len(curve_pts) - 1):
                cv2.line(img, curve_pts[i], curve_pts[i + 1], self.BORDER_COLOR, 2, cv2.LINE_AA)
            for pt in curve_pts:
                cv2.circle(img, pt, 3, self.primary_color, -1, cv2.LINE_AA)

        # X-axis label
        x_label = f"Elapsed Time (Bins of {int(bin_size_sec)}s) ->"
        cv2.putText(img, x_label, (plot_x + plot_w // 2 - 100, plot_y + plot_h + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)

# =============================================================================
# PICTURE GENERATORS (STATIC)
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


class GNGPictureGenerator(PictureGenerator):

    def _format_display_name(self, filename):
        """Converts filename to a display name without '_init' and with spaces instead of '_'."""
        name = os.path.splitext(os.path.basename(filename))[0]

        # Handle the "sym_{from_name}_enables_{to_name}_pass" pattern
        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            from_name = from_name.replace("_", " ").strip()
            to_name = to_name.replace("_", " ").strip()
            return f"{from_name} -> {to_name}"

        if name.endswith("_init"):
            name = name[:-5]
            name += ' - Precondition'
        elif name.endswith("_eff"):
            name = name[:-4]
            name += ' - Effect'
        elif name.endswith("_pass"):
            name = name[:-5]
            name += ' - Transition'
        return name.replace("_", " ").strip()

    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
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
                
                display_title = self._format_display_name(json_path)
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
                file_base = os.path.splitext(os.path.basename(json_path))[0]
                out_path = os.path.join(out_dir, f"{file_base}_invariants.png")
                cv2.imwrite(out_path, img)
                print(f"[PictureGenerator] Saved GNG invariants to {out_path}")
            except Exception as e:
                print(f"[PictureGenerator] Failed to process {json_path}: {e}")

class RepresentativePointPictureGenerator(PictureGenerator):
    """Generates a static sensimotor visualization of the representative medoid point for each symbol JSON."""
    
    def _format_display_name(self, filename):
        """Converts filename to a display name without '_init' and with spaces instead of '_'."""
        name = os.path.splitext(os.path.basename(filename))[0]

        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            from_name = from_name.replace("_", " ").strip()
            to_name = to_name.replace("_", " ").strip()
            return f"{from_name} -> {to_name}"

        if name.endswith("_init"):
            name = name[:-5]
            name += ' - Precondition'
        elif name.endswith("_eff"):
            name = name[:-4]
            name += ' - Effect'
        elif name.endswith("_pass"):
            name = name[:-5]
            name += ' - Transition'
        return name.replace("_", " ").strip()

    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session:
            return

        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.exists(symbols_dir):
            return

        out_dir = self.get_pictures_dir()
        if not out_dir:
            return

        for json_path in glob.glob(os.path.join(symbols_dir, "*.json")):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                nodes = np.array(data.get("nodes", []))
                if len(nodes) == 0:
                    continue

                # Compute medoid (node closest to the centroid)
                centroid = np.mean(nodes, axis=0)
                dists = np.linalg.norm(nodes - centroid, axis=1)
                rep_idx = int(np.argmin(dists))
                rep_node = nodes[rep_idx]

                file_base = os.path.splitext(os.path.basename(json_path))[0]
                display_title = self._format_display_name(json_path)
                img = self.renderer.create_base_canvas(
                    f"REPRESENTATIVE POINT: {display_title}",
                    f"Medoid Node index {rep_idx} out of {len(nodes)} points",
                )

                # Draw LIDAR (dimensions 0:8)
                lidar_data = rep_node[:8]
                self.renderer.draw_lidar(img, lidar_data, is_live=True)

                # Draw Vision Grids (dimensions 8:248)
                cam_data = rep_node[8:248]
                self.renderer.draw_grids(img, cam_data, is_live=True)

                self.renderer.draw_sample_badge(img, n_points=len(nodes))

                out_path = os.path.join(out_dir, f"{file_base}_rep_point.png")
                cv2.imwrite(out_path, img)
                print(
                    f"[Picture Generator] Saved image to {out_path}"
                )

            except Exception as e:
                print(
                    f"[Picture Generator] Error processing {json_path}: {e}"
                )

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
                
            img = self.renderer.create_base_canvas("TRANSITION GRAPH")
            self.renderer.draw_transition_graph(img, graph_data)

            out_dir = self.get_pictures_dir()
            if not out_dir:
                return
            out_path = os.path.join(out_dir, "trans_graph_visualised.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved Transition Graph image to {out_path}")
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate TransGraph: {e}")


class UnifiedGNGPictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", pred_type='init', session_dir=None):
        super().__init__(logs_root, session_dir=session_dir)
        self.gng_colors = {}
        self.pred_type = pred_type
        # Borrowing the distinct, vibrant BGR color palette from the live monitor
        self.color_palette = [
            (60, 60, 220),   (60, 220, 60),   (220, 100, 60),
            (60, 200, 220),  (200, 60, 200),  (220, 200, 60),
            (100, 120, 255), (255, 120, 100), (100, 255, 120),
            (40, 140, 240),  # Orange-ish
            (180, 50, 130),  # Purple / Indigo
            (60, 140, 20),  # Dark / Olive Green
            (160, 130, 240),  # Coral / Salmon
            (150, 210, 240),  # Amber / Gold
        ]

    def _format_display_name(self, filename):
        """Converts filename to a display name without '_init' and with spaces instead of '_'."""
        name = os.path.splitext(os.path.basename(filename))[0]

        # Handle the "sym_{from_name}_enables_{to_name}_pass" pattern
        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            from_name = from_name.replace("_", " ").strip()
            to_name = to_name.replace("_", " ").strip()
            return f"{from_name} -> {to_name}"

        # Default fallback handling
        if name.endswith("_init"):
            name = name[:-5]
        elif name.endswith("_eff"):
            name = name[:-4]
        elif name.endswith("_pass"):
            name = name[:-5]

        return name.replace("_", " ").strip()

    def _get_color(self, predicate_name):
        if predicate_name not in self.gng_colors:
            color_idx = len(self.gng_colors) % len(self.color_palette)
            self.gng_colors[predicate_name] = self.color_palette[color_idx]
        return self.gng_colors[predicate_name]

    def run(self):
        latest_session = self.get_latest_session()
        if not latest_session: return
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.exists(symbols_dir): return

        json_files = glob.glob(os.path.join(symbols_dir, f"*_{self.pred_type}.json"))
        
        G = nx.Graph()
        node_colors = {}
        edge_colors = {}
        
        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                pred_name = self._format_display_name(json_path)
                color = self._get_color(pred_name)
                
                nodes = data.get("nodes", [])
                radiuses = data.get("local_radiuses", [])
                
                for i in range(len(nodes)):
                    node_id = f"{pred_name}_{i}"
                    
                    # Safely map the local radius to the node
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
                print(f"[PictureGenerator] Failed to process {json_path} for unified graph: {e}")

        if len(G.nodes) == 0: return
        pred_type = 'Unknown Type'
        if self.pred_type == 'eff':
            pred_type = 'effects'
        elif self.pred_type == 'init':
            pred_type = 'preconditions'
        elif self.pred_type == 'pass':
            pred_type = "transitions"
        img = self.renderer.create_base_canvas("UNIFIED GNG PLANE", f"Topological state space mapping for {pred_type}")
        self.renderer.draw_unified_GNG(img, G, node_colors, edge_colors, self.gng_colors)

        out_dir = self.get_pictures_dir()
        if not out_dir: return
        out_path = os.path.join(out_dir, f"unified_gng_plane_{self.pred_type}.png")
        cv2.imwrite(out_path, img)
        print(f"[PictureGenerator] Saved Unified GNG plane to {out_path}")
            
class PipelinePictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)
        self.generators = [
            GNGPictureGenerator(logs_root, session_dir=session_dir),
            TransGraphPictureGenerator(logs_root, session_dir=session_dir),
            UnifiedGNGPictureGenerator(logs_root, 'init', session_dir=session_dir),
            UnifiedGNGPictureGenerator(logs_root, 'eff', session_dir=session_dir),
            UnifiedGNGPictureGenerator(logs_root, 'pass', session_dir=session_dir),
            UpdateFrequencyPictureGenerator(logs_root, session_dir=session_dir),
            RepresentativePointPictureGenerator(logs_root, session_dir=session_dir)
        ]

    def run(self):
        for gen in self.generators:
            gen.run()

# =============================================================================
# LIVE MONITORING 
# =============================================================================

class LiveSensimotorMonitor(Renderer):
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

class LiveUpdateFrequencyMonitor(Renderer):
    def __init__(self, logs_root="logs", timestamp_filename="timestamps.txt", bin_size_sec=10.0):
        super().__init__()
        self.logs_root = logs_root
        self.timestamp_filename = timestamp_filename
        self.bin_size_sec = bin_size_sec
        self._last_mtime = 0.0
        self.window_name = "Live Update Frequency Monitor"
        self._window_created = False

    def get_latest_session(self):
        if not os.path.exists(self.logs_root): return None
        session_dirs = [os.path.join(self.logs_root, d) for d in os.listdir(self.logs_root) if os.path.isdir(os.path.join(self.logs_root, d))]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("LIVE UPDATE FREQUENCY", "Waiting for timestamp data...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        latest_session = self.get_latest_session()
        if not latest_session: return

        times_path = os.path.join(latest_session, "graphs", self.timestamp_filename)

        if not os.path.exists(times_path) or os.path.getsize(times_path) == 0:
            return


        try:
            mtime = os.path.getmtime(times_path)
            if mtime > self._last_mtime:
                with open(times_path, "r") as f:
                    timestamps = [float(line.strip()) for line in f if line.strip()]

                img = self.create_base_canvas("LIVE UPDATE FREQUENCY", f"Tracking: {times_path}")
                self.draw_frequency_plot(img, timestamps, bin_size_sec=self.bin_size_sec)
                self.draw_sample_badge(img, n_points=len(timestamps))

                cv2.imshow(self.window_name, img)
                self._last_mtime = mtime
        except Exception as e:
            print(f"[LiveUpdateFrequencyMonitor] Error loading timestamps: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

class LiveGraphMonitor(Renderer):
    def __init__(self, logs_root="logs", trans_filename="trans_graph.json"):
        super().__init__()
        self.logs_root = logs_root
        self.trans_filename = trans_filename
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

class LiveGNGMonitor(Renderer):
    def __init__(self, logs_root="logs"):
        super().__init__()
        self.logs_root = logs_root

        self.window_name = "Live GNG Plane"
        self._window_created = False

        self._last_mtimes = {}
        self.gng_colors = {}
        
        self.color_palette = [
            (60, 60, 220),  # Red-ish
            (60, 220, 60),  # Green-ish
            (220, 100, 60),  # Blue-ish
            (60, 200, 220),  # Yellow-ish
            (200, 60, 200),  # Magenta-ish
            (220, 200, 60),  # Cyan-ish
            (100, 120, 255),  # Light Red
            (255, 120, 100),  # Light Blue
            (100, 255, 120),  # Light Green
            (40, 140, 240),  # Orange-ish
            (180, 50, 130),  # Purple / Indigo
            (60, 140, 20),  # Dark / Olive Green
            (160, 130, 240),  # Coral / Salmon
            (150, 210, 240),  # Amber / Gold
        ]

    def _format_display_name(self, filename):
        """Converts filename to a display name without '_init' and with spaces instead of '_'."""
        name = os.path.splitext(os.path.basename(filename))[0]

        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            from_name = from_name.replace("_", " ").strip()
            to_name = to_name.replace("_", " ").strip()
            return f"{from_name} -> {to_name}"

        if name.endswith("_init"):
            name = name[:-5]
            name += ' - Precondition'
        elif name.endswith("_eff"):
            name = name[:-4]
            name += ' - Effect'
        elif name.endswith("_pass"):
            name = name[:-5]
            name += ' - Transition'
        return name.replace("_", " ").strip()

    def get_latest_session(self):
        if not os.path.exists(self.logs_root):
            return None
        session_dirs = [
            os.path.join(self.logs_root, d)
            for d in os.listdir(self.logs_root)
            if os.path.isdir(os.path.join(self.logs_root, d))
        ]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("LIVE GNG PLANE", "Waiting for predicate data...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def _get_color(self, label):
        if label not in self.gng_colors:
            color_idx = len(self.gng_colors) % len(self.color_palette)
            self.gng_colors[label] = self.color_palette[color_idx]
        return self.gng_colors[label]

    def _load_json_with_retry(self, path, retries=3, delay=0.05):
        for _ in range(retries):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (PermissionError, OSError, json.JSONDecodeError):
                time.sleep(delay)
        return None

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        latest_session = self.get_latest_session()
        if not latest_session:
            return

        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.exists(symbols_dir):
            return

        json_files = glob.glob(os.path.join(symbols_dir, "*_init.json"))
        if not json_files:
            return

        # Check if any symbol file has been modified
        needs_update = False
        for f in json_files:
            try:
                mtime = os.path.getmtime(f)
                if self._last_mtimes.get(f, 0.0) < mtime:
                    needs_update = True
                    self._last_mtimes[f] = mtime
            except OSError:
                continue

        if not needs_update:
            return

        G = nx.Graph()
        node_colors = {}
        edge_colors = {}

        for json_path in json_files:
            data = self._load_json_with_retry(json_path)
            if data is None:
                continue

            display_name = self._format_display_name(json_path)
            color = self._get_color(display_name)

            nodes = data.get("nodes", [])
            radiuses = data.get("local_radiuses", [])

            for i in range(len(nodes)):
                node_id = f"{display_name}_{i}"
                r_val = radiuses[i] if i < len(radiuses) else 0.0
                G.add_node(node_id, radius=r_val)
                node_colors[node_id] = color

            edges_dict = data.get("edges", {})
            for edge_str in edges_dict.keys():
                u_str, v_str = edge_str.split(",")
                u_id = f"{display_name}_{u_str}"
                v_id = f"{display_name}_{v_str}"
                G.add_edge(u_id, v_id)
                edge_colors[(u_id, v_id)] = color
                edge_colors[(v_id, u_id)] = color

        if len(G.nodes) == 0:
            return

        img = self.create_base_canvas("LIVE GNG PLANE", "preconditions")
        # self.gng_colors uses display_name keys (cleaned up for the legend)
        self.draw_unified_GNG(img, G, node_colors, edge_colors, self.gng_colors)
        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

class LiveRepresentativePointMonitor(Renderer):
    """Monitors PDDL symbol JSON files in the latest session and displays the

    representative medoid point for the current active action.
    """

    def __init__(self, bus, logs_root="logs"):
        super().__init__()
        self.bus = bus
        self.logs_root = logs_root
        self.window_name = "Live Representative Point Monitor"
        self._window_created = False

    def get_latest_session(self):
        if not os.path.exists(self.logs_root):
            return None
        session_dirs = [
            os.path.join(self.logs_root, d)
            for d in os.listdir(self.logs_root)
            if os.path.isdir(os.path.join(self.logs_root, d))
        ]
        return max(session_dirs, key=os.path.basename) if session_dirs else None

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas(
                "LIVE REPRESENTATIVE POINT", "Waiting for symbol data..."
            )
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def _resolve_action_name(self):
        try:
            res = self.bus.call_service("agent/ask/action")
            if isinstance(res, dict):
                return res.get("action") or res.get("name")
            return res
        except Exception as e:
            print(f"[LiveRepresentativePointMonitor] Failed to query action service: {e}")
            return None

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        action_name = self._resolve_action_name()
        if not action_name:
            return

        latest_session = self.get_latest_session()
        if not latest_session:
            return

        action_file = os.path.join(
            latest_session, "PDDL", "GNG", f"{action_name}_init.json"
        )
        if not os.path.isfile(action_file):
            return

        try:
            with open(action_file, "r") as f:
                data = json.load(f)

            nodes = np.array(data.get("nodes", []))
            if len(nodes) == 0:
                return

            # Compute medoid
            centroid = np.mean(nodes, axis=0)
            dists = np.linalg.norm(nodes - centroid, axis=1)
            rep_idx = int(np.argmin(dists))
            rep_node = nodes[rep_idx]

            display_title = str(action_name).replace("_", " ")
            img = self.create_base_canvas(
                f"REPRESENTATIVE POINT: {display_title}",
                f"Node {rep_idx} of {len(nodes)} (Medoid)",
            )

            # Dimensions 0:8 for LIDAR and 8:248 for Retinal Ganglion + Cone Cells
            self.draw_lidar(img, rep_node[:8], is_live=True)
            self.draw_grids(img, rep_node[8:248], is_live=True)
            self.draw_sample_badge(img, n_points=len(nodes))

            cv2.imshow(self.window_name, img)

        except Exception as e:
            print(f"[LiveRepresentativePointMonitor] Error rendering point: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

if __name__ == "__main__":
    runner = PipelinePictureGenerator(logs_root="logs") #, session_dir="2026-09-10_19-07")
    runner.run()


