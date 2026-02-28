import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import os
from shapely.geometry import LineString

ox.settings.log_console = True
ox.settings.use_cache = True

def generate_lehigh_nodes_map():
    print("Generating node ID map for Lehigh University...")
    point = (40.6053, -75.3780)
    dist = 800
    
    G = ox.graph_from_point(point, dist=dist, network_type='drive_service')
    try:
        G_undirected = ox.utils_graph.get_undirected(G)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G)
        
    # User specifically requested to cut the southern extension dangling below node 101614211...
    # The exact bottleneck bridge is edge (259302817, 478228138)
    u_cut, v_cut = 259302817, 478228138
    u_del = 1016142119
    
    if G_undirected.has_edge(u_cut, v_cut):
        G_copy = G_undirected.copy()
        G_copy.remove_edge(u_cut, v_cut)
        comp_v = list(nx.node_connected_component(G_copy, v_cut))
        G_undirected.remove_nodes_from(comp_v)
        print(f"Manual surgery: removed southern component of {len(comp_v)} nodes")
        
    if u_del in G_undirected:
        G_undirected.remove_node(u_del)
        print(f"Manual surgery: removed user target node {u_del}")

    # Additional User Cuts:
    # 1. Edge 6204650796 <-> 6204650804
    if G_undirected.has_edge(6204650796, 6204650804):
        G_undirected.remove_edge(6204650796, 6204650804)
    
    # 2. Node 259317685
    if 259317685 in G_undirected:
        G_undirected.remove_node(259317685)
        
    # 3. Node 248551974 and right loop
    right_loop_nodes = {248551974, 1016142066, 1016377132, 1016376957, 1016377134}
    G_undirected.remove_nodes_from([n for n in right_loop_nodes if n in G_undirected])
    
    # 4. Node 1121706745 and left building triangle/circle
    left_loop_nodes = {259368711, 472425550, 259257425, 1121706772, 1121706745, 258574525, 259274110}
    G_undirected.remove_nodes_from([n for n in left_loop_nodes if n in G_undirected])

    # 5. Connect 335361039 and 335360953
    if 335361039 in G_undirected and 335360953 in G_undirected:
        u_edge, v_edge = 335361039, 335360953
        x1, y1 = G_undirected.nodes[u_edge]['x'], G_undirected.nodes[u_edge]['y']
        x2, y2 = G_undirected.nodes[v_edge]['x'], G_undirected.nodes[v_edge]['y']
        geom = LineString([(x1, y1), (x2, y2)])
        length = ox.distance.great_circle(y1, x1, y2, x2)
        G_undirected.add_edge(u_edge, v_edge, key=0, length=length, geometry=geom)

    # Baseline Prune dead ends
    while True:
        nodes_to_remove = []
        for n, d in G_undirected.degree():
            if d < 2:
                nodes_to_remove.append(n)
        if not nodes_to_remove:
            break
        G_undirected.remove_nodes_from(nodes_to_remove)
        
    if len(G_undirected) > 0:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()
        
    os.makedirs('Map', exist_ok=True)
    out_file = "Map/lehigh_map_nodes.png"
    
    tags = {"building": True}
    try:
        buildings = ox.features.features_from_point(point, tags=tags, dist=dist)
    except AttributeError:
        buildings = ox.geometries.geometries_from_point(point, tags=tags, dist=dist)

    fig, ax = plt.subplots(figsize=(24, 24))
    
    ox.plot_graph(
        G_undirected,
        ax=ax,
        show=False,
        save=False,
        node_size=30,
        node_color='red',
        edge_color='lightblue',
        bgcolor='whitesmoke'
    )
    
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor='lightgray', edgecolor='darkgray', zorder=1, alpha=0.8)
    
    # 添加节点 ID 标签
    pos = {n: (d['x'], d['y']) for n, d in G_undirected.nodes(data=True)}
    labels = {n: str(n) for n in G_undirected.nodes()}
    
    # 画出带白色发光轮廓的黑色文字
    nx.draw_networkx_labels(
        G_undirected, 
        pos, 
        labels, 
        font_size=6, 
        font_color='black', 
        font_weight='bold',
        ax=ax, 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2)
    )
                    
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    generate_lehigh_nodes_map()
