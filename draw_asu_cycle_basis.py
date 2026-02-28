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
# Step 1: Reconstruct the ASU pruned graph
# ─────────────────────────────────────────────
def build_asu_graph():
    point = (33.3055, -111.6815)
    G = ox.graph_from_point(point, dist=800, network_type='drive_service')
    try:
        G_undirected = ox.utils_graph.get_undirected(G)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G)

    # Pre-consolidation surgery: remove 1649* nodes
    nodes_1649 = [n for n in G_undirected.nodes() if str(n).startswith("1649")]
    G_undirected.remove_nodes_from(nodes_1649)

    # Remove branch starting at 5259118345
    n_5259 = 5259118345
    n_8310 = 8310002397
    if n_5259 in G_undirected:
        queue = [n_5259]
        visited = set([n_5259, n_8310])
        sub = {n_5259}
        while queue:
            curr = queue.pop(0)
            for nbr in G_undirected.neighbors(curr):
                if nbr not in visited:
                    visited.add(nbr)
                    sub.add(nbr)
                    queue.append(nbr)
        G_undirected.remove_nodes_from(sub)

    # Consolidate intersections
    G_proj = ox.project_graph(G_undirected)
    G_cons_proj = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)
    G_und_dir = ox.project_graph(G_cons_proj, to_crs="EPSG:4326")
    try:
        G_undirected = ox.utils_graph.get_undirected(G_und_dir)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G_und_dir)

    # Post-consolidation custom surgery
    # Delete edge 19-102 & its loop
    if G_undirected.has_edge(19, 102):
        G_undirected.remove_edge(19, 102)
        if 102 in G_undirected:
            comp = list(nx.node_connected_component(G_undirected, 102))
            G_undirected.remove_nodes_from(comp)
    # Delete edge 28-29 & its loop
    if G_undirected.has_edge(28, 29):
        G_undirected.remove_edge(28, 29)
        if 29 in G_undirected:
            comp = list(nx.node_connected_component(G_undirected, 29))
            G_undirected.remove_nodes_from(comp)

    # Add straight-line edges: 15-13, 100-98, 13-31, 31-32
    for u, v in [(15, 13), (100, 98), (13, 31), (31, 32)]:
        if u in G_undirected and v in G_undirected:
            x1, y1 = G_undirected.nodes[u]['x'], G_undirected.nodes[u]['y']
            x2, y2 = G_undirected.nodes[v]['x'], G_undirected.nodes[v]['y']
            geom = LineString([(x1, y1), (x2, y2)])
            length = ox.distance.great_circle(y1, x1, y2, x2)
            G_undirected.add_edge(u, v, key=0, length=length, geometry=geom)

    # Dead-end pruning
    x_max = -111.670
    preserve_bbox = (33.3047, 33.3095, -111.6808, -111.6720)
    def in_preserve_box(lat, lon):
        south, north, west, east = preserve_bbox
        return (south <= lat <= north) and (west <= lon <= east)

    while True:
        nodes_to_remove = []
        for n, d in G_undirected.degree():
            if d < 2:
                nd = G_undirected.nodes[n]
                lat = nd.get('y', nd.get('lat', 0))
                lon = nd.get('x', nd.get('lon', 0))
                if not in_preserve_box(lat, lon):
                    nodes_to_remove.append(n)
        for n in list(G_undirected.nodes()):
            if n not in nodes_to_remove:
                nd = G_undirected.nodes[n]
                lat = nd.get('y', nd.get('lat', 0))
                lon = nd.get('x', nd.get('lon', 0))
                if lon > x_max and not in_preserve_box(lat, lon):
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
    print("Building ASU road network graph...")
    G_undirected = build_asu_graph()
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
        f"ASU Polytechnic Campus – Oriented Cycle Basis ({k} cycles)\n"
        f"Green = +1 (forward), Red = −1 (reverse)",
        fontsize=15, fontweight="bold"
    )

    # Panel 0: the full graph
    ax0 = axes[0]
    ax0.set_title("Full ASU Road Graph", fontsize=9, fontweight="bold")
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
    out_path = "visuals/asu_cycle_basis_panel.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
