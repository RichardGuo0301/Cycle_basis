import osmnx as ox
import matplotlib.pyplot as plt
import os
import math
from shapely.geometry import LineString

# 配置参数
ox.settings.log_console = True
ox.settings.use_cache = True

def draw_and_save_map(point, dist, filename, title, network_type='all', x_max=None, prune_dead_ends=True, apply_custom_surgery=False, preserve_bbox=None):
    import networkx as nx
    print(f"\n{'='*50}")
    print(f"正在下载: {title} ...")
    
    # 获取网络图
    G = ox.graph_from_point(point, dist=dist, network_type=network_type)
    try:
        G_undirected = ox.utils_graph.get_undirected(G)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G)
        
    # === CUSTOM MANUAL SURGERY ===
    if apply_custom_surgery:
        if "ASU" in title:
            # 1. Remove all 1649* nodes
            nodes_1649 = [n for n in G_undirected.nodes() if str(n).startswith("1649")]
            G_undirected.remove_nodes_from(nodes_1649)
            print(f"[{title}] Manual surgery: removed {len(nodes_1649)} nodes starting with 1649*")
        
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
                print(f"[{title}] Manual surgery: removed {len(subgraph_nodes)} nodes attached to {n_5259}")
                
        elif "Lehigh" in title:
            # User specifically requested to cut the southern extension dangling below node 101614211...
            # The exact bottleneck bridge is edge (259302817, 478228138)
            u_cut, v_cut = 259302817, 478228138
            u_del = 1016142119
            
            if G_undirected.has_edge(u_cut, v_cut):
                G_copy = G_undirected.copy()
                G_copy.remove_edge(u_cut, v_cut)
                comp_v = list(nx.node_connected_component(G_copy, v_cut))
                G_undirected.remove_nodes_from(comp_v)
                print(f"[{title}] Manual surgery: removed southern component of {len(comp_v)} nodes")
                
            if u_del in G_undirected:
                G_undirected.remove_node(u_del)
                
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
            
            print(f"[{title}] Manual surgery: applied 4 specific user cuts (loops, isolated disconnected edges) and 1 custom connection")
    # =============================
        
    # 如果指定了 x_max，剪裁掉靠东(右)侧的节点
    # 但是，如果指定了 preserve_bbox，我们不应该单纯切掉 x_max 右侧的
    if x_max is not None and preserve_bbox is None:
        nodes_to_remove = [n for n, d in G_undirected.nodes(data=True) if d['x'] > x_max]
        G_undirected.remove_nodes_from(nodes_to_remove)
        
    def in_preserve_box(lat, lon):
        if preserve_bbox is None: return False
        south, north, west, east = preserve_bbox
        return (south <= lat <= north) and (west <= lon <= east)

    # Consolidate intersections if we have a preserve box to reduce density there
    if preserve_bbox is not None:
        print(f"[{title}] Consolidating intersections to reduce density in extended region...")
        G_proj = ox.project_graph(G_undirected)
        G_cons_proj = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)
        G_und_dir = ox.project_graph(G_cons_proj, to_crs="EPSG:4326")
        try:
            G_undirected = ox.utils_graph.get_undirected(G_und_dir)
        except AttributeError:
            G_undirected = ox.convert.to_undirected(G_und_dir)
            
    if apply_custom_surgery and "ASU" in title:
        if G_undirected.has_edge(19, 102):
            G_undirected.remove_edge(19, 102)
            if 102 in G_undirected:
                comp_102 = list(nx.node_connected_component(G_undirected, 102))
                G_undirected.remove_nodes_from(comp_102)
        if G_undirected.has_edge(28, 29):
            G_undirected.remove_edge(28, 29)
            if 29 in G_undirected:
                comp_29 = list(nx.node_connected_component(G_undirected, 29))
                G_undirected.remove_nodes_from(comp_29)
                
        edges_to_add = [(15, 13), (100, 98), (13, 31), (31, 32)]
        for u, v in edges_to_add:
            if u in G_undirected and v in G_undirected:
                x1, y1 = G_undirected.nodes[u]['x'], G_undirected.nodes[u]['y']
                x2, y2 = G_undirected.nodes[v]['x'], G_undirected.nodes[v]['y']
                geom = LineString([(x1, y1), (x2, y2)])
                length = ox.distance.great_circle(y1, x1, y2, x2)
                G_undirected.add_edge(u, v, key=0, length=length, geometry=geom)

    # 核心修剪：剔除死胡同枝杈 (Degree < 2 不存在于任何循环)
    if prune_dead_ends:
        while True:
            nodes_to_remove = []
            for n, d in G_undirected.degree():
                if d < 2:
                    # Sparse dead-ends if they are inside the preserve bounding box
                    node_data = G_undirected.nodes[n]
                    if 'y' in node_data and 'x' in node_data:
                        lat, lon = node_data['y'], node_data['x']
                    elif 'lat' in node_data and 'lon' in node_data:
                        lat, lon = node_data['lat'], node_data['lon']
                    else:
                        nodes_to_remove.append(n)
                        continue
                        
                    if not in_preserve_box(lat, lon):
                        # x_max constraint combined with preserve_bbox:
                        # If node is outside preserve box, AND it is > x_max, delete it anyway even if degree >= 2
                        nodes_to_remove.append(n)
            
            # Additional sweep for nodes strictly outside x_max and outside preserve_bbox
            if x_max is not None and preserve_bbox is None:
                pass # Already handled earlier
            elif x_max is not None and preserve_bbox is not None:
                for n in list(G_undirected.nodes()):
                    if n not in nodes_to_remove:
                        node_data = G_undirected.nodes[n]
                        lat = node_data.get('y', node_data.get('lat', 0))
                        lon = node_data.get('x', node_data.get('lon', 0))
                        if lon > x_max and not in_preserve_box(lat, lon):
                            nodes_to_remove.append(n)

            if not nodes_to_remove:
                break
            G_undirected.remove_nodes_from(nodes_to_remove)
    
    # 只保留最大连通图
    if len(G_undirected) > 0:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()
        
    print(f"[{title}] 网络处理成功: {G_undirected.number_of_nodes()} 节点, {G_undirected.number_of_edges()} 边")
    
    # 获取建筑物
    print(f"[{title}] 正在获取建筑物背景...")
    tags = {"building": True}
    try:
        buildings = ox.features.features_from_point(point, tags=tags, dist=dist)
    except AttributeError:
        buildings = ox.geometries.geometries_from_point(point, tags=tags, dist=dist)
    
    import os
    os.makedirs('Map', exist_ok=True)
    output_file = f"Map/{filename}"
    print(f"[{title}] 正在绘图并保存到 {output_file} ...")
    
    fig, ax = ox.plot_graph(
        G_undirected,
        show=False,
        save=False,
        node_size=25,
        node_color='red',
        node_edgecolor='none',
        node_zorder=4,
        edge_color='royalblue',
        edge_linewidth=2.5,
        edge_alpha=0.9,
        bgcolor='whitesmoke'
    )
    
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor='lightgray', edgecolor='darkgray', zorder=1, alpha=0.8)
    
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[{title}] 保存完毕！")

# 1. ASU Polytechnic Campus (剪枝和过滤版)
# 偏移中心向西移动，使用 drive_service 过滤极其密集的人行小路，并应用 k_core 剪枝
draw_and_save_map(
    point=(33.3055, -111.6815), 
    dist=800, 
    x_max=-111.670,  # 截断东侧接近机场跑道的网格
    network_type='drive_service', 
    apply_custom_surgery=True,
    preserve_bbox=(33.3047, 33.3095, -111.6808, -111.6720), # (south, north, west, east)
    filename="asu_poly_map.png", 
    title="ASU Polytechnic Campus (Pruned)"
)

# 2. Lehigh University (宾夕法尼亚州, 依山而建)
draw_and_save_map(
    point=(40.6053, -75.3780), 
    dist=800, 
    network_type='drive_service',
    apply_custom_surgery=True,
    filename="lehigh_university_map.png", 
    title="Lehigh University (Pruned)"
)

# 3. ASU Polytechnic Campus (完整版)
# 完全保留所有的末端小路、人行道和停机坪道路
draw_and_save_map(
    point=(33.3055, -111.6775), 
    dist=800, 
    network_type='all',
    prune_dead_ends=False,
    filename="asu_poly_map_complete.png", 
    title="ASU Polytechnic Campus (Complete)"
)
