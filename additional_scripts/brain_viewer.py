import glob
import os
import matplotlib.pyplot as plt
import networkx as nx


def find_newest_graph_file(logs_dir="logs"):
    """Finds the most recent transgraph_final.graphml file based on the date-stamped folder name."""
    pattern = os.path.join(
        logs_dir, "*", "brain_network", "transgraph_final.graphml"
    )
    matching_files = glob.glob(pattern)

    if not matching_files:
        return None

    # Sorts folder names formatted like 'YYYYMMDD_HHMM' chronologically
    return sorted(matching_files)[-1]


def run_viewer(file_path=None):
    # Auto-resolve the latest file if none is explicitly provided
    if file_path is None:
        file_path = find_newest_graph_file()

    if not file_path or not os.path.exists(file_path):
        print(f"No graph file found. Path: {file_path}")
        return

    print(f"Loading graph from: {file_path}")

    # Load the graph
    G = nx.read_graphml(file_path)

    # Set up static figure
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title("Brain Network Viewer")
    ax.set_title(
        f"Final Brain Network (Nodes: {len(G.nodes)} | Edges: {len(G.edges)})"
    )

    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes and edges
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_color="skyblue",
            node_size=400,
            font_size=9,
            font_weight="bold",
            edge_color="gray",
            arrows=True,
        )

        # Draw edge labels if present
        edge_labels = nx.get_edge_attributes(G, "skill")
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, ax=ax, font_size=7, label_pos=0.5
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_viewer()