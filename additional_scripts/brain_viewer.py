import matplotlib.pyplot as plt
import networkx as nx
import os
import time

def run_viewer(file_path="logs/brain_network_final.graphml", refresh_rate=0.5):
    print("Starting Live Brain Viewer...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title("Live Brain Network Viewer")

    last_modified_time = 0

    while True:
        try:
            # Only update if the file actually exists and has been modified
            if os.path.exists(file_path):
                current_time = os.path.getmtime(file_path)
                
                if current_time != last_modified_time:
                    last_modified_time = current_time
                    
                    # Read the graph state saved by the agent
                    G = nx.read_graphml(file_path)
                    
                    ax.clear()
                    ax.set_title(f"Live Brain Network (Nodes: {len(G.nodes)} | Edges: {len(G.edges)})")
                    
                    if len(G.nodes) > 0:
                        pos = nx.spring_layout(G, seed=42)
                        
                        nx.draw(
                            G, pos, ax=ax, 
                            with_labels=True, 
                            node_color='skyblue', 
                            node_size=400, 
                            font_size=9, 
                            font_weight='bold', 
                            edge_color='gray',
                            arrows=True
                        )
                        
                        # Extract edge labels if they exist
                        edge_labels = nx.get_edge_attributes(G, 'skill')
                        if edge_labels:
                            nx.draw_networkx_edge_labels(
                                G, pos, 
                                edge_labels=edge_labels, 
                                ax=ax, 
                                font_size=7,
                                label_pos=0.5
                            )
            
            # Allow matplotlib to render
            plt.pause(refresh_rate)
            
        except Exception as e:
            # Catch read errors if we try to read exactly while the agent is writing
            plt.pause(refresh_rate)
            continue

if __name__ == "__main__":
    run_viewer()