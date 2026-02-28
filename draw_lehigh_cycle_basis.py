"""
Draw the oriented cycle basis panel for the ASU Polytechnic Campus pruned map.
Uses the same surgery logic as get_warehouse_osmnx.py to reconstruct the graph.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox

# Hotfix for NumPy 2.0 + older NetworkX
if not hasattr(np, 'alltrue'):
    np.alltrue = np.all
from shapely.geometry import LineString

from horton_gram_mcb import find_horton_gram_mcb

# ─────────────────────────────────────────────
# Step 1: Reconstruct the Lehigh pruned graph
# ─────────────────────────────────────────────
def build_lehigh_graph():
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
        
    # 6. Connect 1016142113 and 259346266
    if 1016142113 in G_undirected and 259346266 in G_undirected:
        u_edge, v_edge = 1016142113, 259346266
        x1, y1 = G_undirected.nodes[u_edge]['x'], G_undirected.nodes[u_edge]['y']
        x2, y2 = G_undirected.nodes[v_edge]['x'], G_undirected.nodes[v_edge]['y']
        geom = LineString([(x1, y1), (x2, y2)])
        length = ox.distance.great_circle(y1, x1, y2, x2)
        G_undirected.add_edge(u_edge, v_edge, key=0, length=length, geometry=geom)

    # 7. Connect 60978330 and 60978331
    if 60978330 in G_undirected and 60978331 in G_undirected:
        u_edge, v_edge = 60978330, 60978331
        x1, y1 = G_undirected.nodes[u_edge]['x'], G_undirected.nodes[u_edge]['y']
        x2, y2 = G_undirected.nodes[v_edge]['x'], G_undirected.nodes[v_edge]['y']
        geom = LineString([(x1, y1), (x2, y2)])
        length = ox.distance.great_circle(y1, x1, y2, x2)
        G_undirected.add_edge(u_edge, v_edge, key=0, length=length, geometry=geom)

    # 8. For 1166459 nodes: remove if it's a dead end (degree == 1)
    nodes_116 = [n for n in G_undirected.nodes() if str(n).startswith("1166459")]
    for n in nodes_116:
        if G_undirected.degree(n) == 1:
            G_undirected.remove_node(n)

    # 9. Unconditionally remove specific 1166459* nodes that the user requested
    for n in [11664593244, 11664593242, 11664593245, 11664593249]:
        if n in G_undirected:
            G_undirected.remove_node(n)

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

    return G_undirected


# ─────────────────────────────────────────────
# Step 2: Cycle basis + oriented matrix
# ─────────────────────────────────────────────
def build_oriented_cycle_matrix(G_undirected, edge_list, basis_meta):
    m = len(edge_list)
    k = len(basis_meta)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}

    C_matrix = np.zeros((m, k), dtype=int)
    for j, c_data in enumerate(basis_meta):
        for i in range(m):
            if (c_data["bv"] >> i) & 1:
                C_matrix[i, j] = 1

    C_oriented = np.zeros((m, k), dtype=int)
    for j, c_data in enumerate(basis_meta):
        cycle_edges = [edge_list[i] for i in range(m) if C_matrix[i, j] == 1]
        cycle_subgraph = nx.Graph(cycle_edges)
        try:
            cycle_nodes = [e[0] for e in nx.find_cycle(cycle_subgraph)]
        except Exception:
            cycle_nodes = list(cycle_subgraph.nodes())

        for i in range(len(cycle_nodes)):
            u_curr = cycle_nodes[i]
            v_curr = cycle_nodes[(i + 1) % len(cycle_nodes)]
            u, v = sorted((u_curr, v_curr))
            if (u, v) in edge_to_idx:
                idx = edge_to_idx[(u, v)]
                C_oriented[idx, j] = +1 if (u_curr == u and v_curr == v) else -1

    return C_oriented


# ─────────────────────────────────────────────
# Step 3: Render
# ─────────────────────────────────────────────
def main():
    print("Building Lehigh University road network graph...")
    G_undirected = build_lehigh_graph()
    n = G_undirected.number_of_nodes()
    m_edges = G_undirected.number_of_edges()
    print(f"Graph ready: n={n}, m={m_edges}")

    # ── Print travel time table (speed = 16 m/s assumed) ──────────────────
    SPEED = 16.0   # m/s
    print()
    print(f"{'Edge':>4}  {'u':>6}  {'v':>6}  {'Length(m)':>10}  {'Time(s)':>8}")
    print("-" * 40)
    total_length = 0.0
    edge_list_raw = sorted([tuple(sorted((u, v))) for u, v in G_undirected.edges()])
    edge_list_raw = list(dict.fromkeys(edge_list_raw))
    for idx, (u, v) in enumerate(edge_list_raw):
        edge_data = G_undirected[u][v]
        data = edge_data[0] if 0 in edge_data else next(iter(edge_data.values()))
        length = data.get('length', 0.0)
        travel_time = length / SPEED
        total_length += length
        print(f"{idx:>4}  {u:>6}  {v:>6}  {length:>10.1f}  {travel_time:>8.1f}")
    print("-" * 40)
    print(f"{'TOTAL':>4}  {'':>6}  {'':>6}  {total_length:>10.1f}  {total_length/SPEED:>8.1f}")
    print()


    pos = {node: (data['x'], data['y']) for node, data in G_undirected.nodes(data=True)}

    edge_list = sorted([tuple(sorted((u, v))) for u, v in G_undirected.edges()])
    # Deduplicate (multigraph may have parallel edges)
    edge_list = list(dict.fromkeys(edge_list))

    print("Computing Horton-Gram MCB...")
    basis_meta, edge_list = find_horton_gram_mcb(G_undirected, edge_list)
    C_oriented = build_oriented_cycle_matrix(G_undirected, edge_list, basis_meta)
    k = C_oriented.shape[1]
    print(f"Found {k} cycle basis vectors.")

    # Layout: 4 columns
    ncols = 5
    n_panels = k + 1
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), facecolor='#f8f8f8')
    axes = np.array(axes).reshape(-1)

    fig.suptitle(
        f"Lehigh University – Oriented Cycle Basis ({k} cycles)\n"
        f"Green = +1 (forward), Red = −1 (reverse)",
        fontsize=15, fontweight="bold"
    )

    # Panel 0: the full graph
    ax0 = axes[0]
    ax0.set_title("Full Lehigh Road Graph", fontsize=9, fontweight="bold")
    nx.draw_networkx_edges(G_undirected, pos, edge_color="#888", width=1.0, alpha=0.6, ax=ax0)
    nx.draw_networkx_nodes(G_undirected, pos, node_color="#4292c6", node_size=60, edgecolors="black", linewidths=0.5, ax=ax0)
    nx.draw_networkx_labels(G_undirected, pos, font_size=5, font_weight="bold", ax=ax0)
    ax0.axis("off")

    # Cycle panels
    for j in range(k):
        ax = axes[j + 1]
        ax.axis("off")

        # Ghost background
        nx.draw_networkx_edges(G_undirected, pos, edge_color="lightgray", width=0.6, alpha=0.2, ax=ax)
        nx.draw_networkx_nodes(G_undirected, pos, node_color="#e0e0e0", node_size=25, edgecolors="none", ax=ax)

        pos_edges, neg_edges, active_nodes = [], [], set()
        for i, (u, v) in enumerate(edge_list):
            val = int(C_oriented[i, j])
            if val == 0:
                continue
            active_nodes.add(u)
            active_nodes.add(v)
            if val > 0:
                pos_edges.append((u, v))
            else:
                neg_edges.append((v, u))

        if pos_edges:
            nx.draw_networkx_edges(
                nx.DiGraph(pos_edges), pos, edgelist=pos_edges,
                edge_color="green", width=1.6, alpha=0.9,
                arrows=True, arrowstyle="-|>", arrowsize=10, ax=ax
            )
        if neg_edges:
            nx.draw_networkx_edges(
                nx.DiGraph(neg_edges), pos, edgelist=neg_edges,
                edge_color="red", width=1.6, alpha=0.9,
                arrows=True, arrowstyle="-|>", arrowsize=10, ax=ax
            )

        if active_nodes:
            sub = G_undirected.subgraph(active_nodes)
            nx.draw_networkx_nodes(sub, pos, node_color="#9ecae1", node_size=100, edgecolors="black", linewidths=0.8, ax=ax)
            nx.draw_networkx_labels(sub, pos, font_size=6, font_weight="bold", ax=ax)

        green_cnt = int(np.sum(C_oriented[:, j] > 0))
        red_cnt = int(np.sum(C_oriented[:, j] < 0))
        ax.set_title(
            f"Cycle {j+1}  (↑{green_cnt} ↓{red_cnt})",
            fontsize=7, fontweight="bold"
        )

    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")

    os.makedirs("visuals", exist_ok=True)
    out_path = "visuals/lehigh_cycle_basis_panel.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
