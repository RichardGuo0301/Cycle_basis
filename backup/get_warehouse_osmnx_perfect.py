import osmnx as ox
import matplotlib.pyplot as plt

# 配置参数
ox.settings.log_console = True
ox.settings.use_cache = True

def draw_and_save_map(point, dist, filename, title, network_type='all', x_max=None, prune_dead_ends=True, apply_custom_surgery=False):
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
    # =============================
        
    # 如果指定了 x_max，剪裁掉靠东(右)侧的节点
    if x_max is not None:
        nodes_to_remove = [n for n, d in G_undirected.nodes(data=True) if d['x'] > x_max]
        G_undirected.remove_nodes_from(nodes_to_remove)
        
    # 核心修剪：剔除死胡同枝杈 (Degree < 2 不存在于任何循环)
    if prune_dead_ends:
        while True:
            nodes_to_remove = [n for n, d in G_undirected.degree() if d < 2]
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
    filename="asu_poly_map.png", 
    title="ASU Polytechnic Campus (Pruned)"
)

# 2. Lehigh University (宾夕法尼亚州, 依山而建)
draw_and_save_map(
    point=(40.6053, -75.3780), 
    dist=800, 
    network_type='drive_service',
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
