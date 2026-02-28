import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import os

ox.settings.log_console = True
ox.settings.use_cache = True

def generate_node_id_map():
    print("Generating node ID map for ASU Pruned...")
    point=(33.3055, -111.6815)
    G = ox.graph_from_point(point, dist=800, network_type='drive_service')
    try:
        G_undirected = ox.utils_graph.get_undirected(G)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G)
    
    # === CUSTOM MANUAL SURGERY ===
    # 1. Remove all 1649* nodes
    nodes_1649 = [n for n in G_undirected.nodes() if str(n).startswith("1649")]
    G_undirected.remove_nodes_from(nodes_1649)
    print(f"Applied manual surgery in guide: removed {len(nodes_1649)} nodes starting with 1649*")

    # 2. Remove branch starting at 5259118345, excluding 8310002397
    n_5259 = 5259118345
    n_8310 = 8310002397
    subgraph_nodes = set()
    if n_5259 in G_undirected:
        queue = [n_5259]
        visited = set([n_5259, n_8310])
        subgraph_nodes.add(n_5259)
        while queue:
            curr = queue.pop(0)
            for nbr in G_undirected.neighbors(curr):
                if nbr not in visited:
                    visited.add(nbr)
                    subgraph_nodes.add(nbr)
                    queue.append(nbr)
        G_undirected.remove_nodes_from(subgraph_nodes)
        print(f"Manual surgery in guide: removed {len(subgraph_nodes)} nodes attached to {n_5259}")
    # =============================
    
    # 模拟 get_warehouse_osmnx.py 中 ASU (Pruned) 的裁剪逻辑
    x_max = -111.670
    nodes_to_remove = [n for n, d in G_undirected.nodes(data=True) if d['x'] > x_max]
    G_undirected.remove_nodes_from(nodes_to_remove)
    
    while True:
        nodes_to_remove = [n for n, d in G_undirected.degree() if d < 2]
        if not nodes_to_remove:
            break
        G_undirected.remove_nodes_from(nodes_to_remove)
        
    if len(G_undirected) > 0:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()

    fig, ax = plt.subplots(figsize=(24, 24)) # 很大的画布
    ox.plot_graph(
        G_undirected,
        ax=ax,
        show=False,
        save=False,
        node_size=30,
        node_color='red',
        edge_color='lightblue',
        edge_linewidth=2,
        bgcolor='whitesmoke'
    )
    
    # 添加节点 ID 标签
    pos = {n: (d['x'], d['y']) for n, d in G_undirected.nodes(data=True)}
    labels = {n: str(n) for n in G_undirected.nodes()}
    
    # 画出带白色发光轮廓的黑色文字，高度可读
    nx.draw_networkx_labels(
        G_undirected, 
        pos, 
        labels, 
        font_size=8, 
        font_color='black', 
        font_weight='bold',
        ax=ax, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3)
    )
    
    plt.savefig('Map/asu_poly_map_nodes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Map/asu_poly_map_nodes.png")

def generate_bbox_map():
    print("Generating Bounding Box map for ASU Complete...")
    point=(33.3055, -111.6775)
    G = ox.graph_from_point(point, dist=800, network_type='all')
    try:
        G_undirected = ox.utils_graph.get_undirected(G)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G)
        
    fig, ax = plt.subplots(figsize=(18, 18))
    
    # Use pure Matplotlib to avoid OSMnx hiding axes
    for u, v, data in G_undirected.edges(data=True):
        x1, y1 = G_undirected.nodes[u]['x'], G_undirected.nodes[u]['y']
        x2, y2 = G_undirected.nodes[v]['x'], G_undirected.nodes[v]['y']
        ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.8, alpha=0.6)
        
    xs = [d['x'] for n, d in G_undirected.nodes(data=True)]
    ys = [d['y'] for n, d in G_undirected.nodes(data=True)]
    ax.scatter(xs, ys, s=15, color='darkcyan', zorder=5)
    ax.set_facecolor('whitesmoke')
    
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(15))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(15))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=12)
    ax.grid(True, linestyle='--', color='black', alpha=0.3)
    
    ax.set_xlabel("Longitude (X coordinate)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Latitude (Y coordinate)", fontsize=16, fontweight='bold')
    ax.set_title("ASU Complete Map - Coordinate Reference Guide", fontsize=20, fontweight='bold', pad=20)
    
    os.makedirs('Map', exist_ok=True)
    plt.savefig('Map/asu_poly_map_complete_bbox.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Map/asu_poly_map_complete_bbox.png")

if __name__ == "__main__":
    generate_node_id_map()
    generate_bbox_map()
