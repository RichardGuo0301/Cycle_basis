"""
Latency-Aware Animation for Method B (Reward-Inverse Node Latency)
Higher-reward nodes get lower latency requirements (more frequent visits).
Only violated nodes get V_MIN_i += 1 after each iteration.
"""
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import milp, LinearConstraint, Bounds
import os

from cycle_basis_demo import create_random_polygon_graph
from horton_gram_mcb import find_horton_gram_mcb
from astar_vmin_search import astar_vmin_search

os.makedirs("visuals", exist_ok=True)


def greedy_minmax_euler_circuit(DG, source, edge_tau_dict, latency_req, node_to_idx):
    """
    Min-Max Greedy Euler circuit with Hierholzer splice fallback.
    - Phase 1: Pure min-max greedy (no bridge detection) — O(E×V), fast
    - Phase 2: If dead end, splice remaining edges via Hierholzer — O(E)
    
    This guarantees a complete Euler circuit every time while keeping the
    greedy-optimized portion as large as possible for latency quality.
    
    Args:
        DG: nx.MultiDiGraph with the directed edges
        source: starting node
        edge_tau_dict: dict mapping (u,v)->travel time
        latency_req: array of per-node latency requirements
        node_to_idx: dict mapping node label -> index in latency_req
    
    Returns:
        list of (u,v) tuples forming the complete Euler circuit
    """
    # Build adjacency with edge keys for tracking
    remaining = {}
    for u in DG.nodes():
        remaining[u] = []
    for u, v, key in DG.edges(keys=True):
        remaining[u].append((v, key))
    
    all_nodes = list(DG.nodes())
    total_edges = DG.number_of_edges()
    path = []
    current = source
    current_time = 0.0
    last_visit_time = {v: 0.0 for v in all_nodes}
    edges_used = 0
    
    # Phase 1: Pure min-max greedy (fast, no bridge detection)
    while edges_used < total_edges:
        neighbors = remaining[current]
        if not neighbors:
            break  # dead end — will splice remaining edges below
        
        if len(neighbors) == 1:
            best_v, best_key = neighbors[0]
        else:
            best_score = float("inf")
            best_v, best_key = None, None
            for v_cand, key_cand in neighbors:
                new_time = current_time + edge_tau_dict.get((current, v_cand), 1.0)
                worst_urg = 0.0
                for node in all_nodes:
                    if node == v_cand:
                        urg = 0.0
                    else:
                        wait = new_time - last_visit_time[node]
                        urg = wait / max(latency_req[node_to_idx[node]], 1e-6)
                    worst_urg = max(worst_urg, urg)
                if worst_urg < best_score:
                    best_score = worst_urg
                    best_v, best_key = v_cand, key_cand
        
        path.append((current, best_v))
        remaining[current].remove((best_v, best_key))
        travel_time = edge_tau_dict.get((current, best_v), 1.0)
        current_time += travel_time
        last_visit_time[best_v] = current_time
        current = best_v
        edges_used += 1
    
    # Phase 2: Splice remaining edges if greedy hit a dead end
    if edges_used < total_edges:
        leftover_count = total_edges - edges_used
        # Build subgraph from remaining edges
        RG = nx.MultiDiGraph()
        for u_r in remaining:
            for (v_r, k_r) in remaining[u_r]:
                RG.add_edge(u_r, v_r, key=k_r)
        
        # Find all weakly connected components with edges
        components = list(nx.weakly_connected_components(RG))
        sub_circuits = []
        for comp in components:
            sub = RG.subgraph(comp).copy()
            if sub.number_of_edges() == 0:
                continue
            # Find a node in this component that also appears in our path
            # so we know where to splice it in
            splice_node = None
            for u_s, v_s in path:
                if v_s in comp:
                    splice_node = v_s
                    break
                if u_s in comp:
                    splice_node = u_s
                    break
            if splice_node is None:
                # Use any node with edges as starting point
                splice_node = next(n for n in comp if sub.out_degree(n) > 0)
            
            try:
                sub_path = list(nx.eulerian_circuit(sub, source=splice_node))
                sub_circuits.append((splice_node, sub_path))
            except (nx.NetworkXError, StopIteration):
                # If Hierholzer fails on this component, skip
                print(f"  [splice] Warning: Hierholzer failed on component with {sub.number_of_edges()} edges")
                continue
        
        # Splice sub-circuits into the main path at junction points
        for splice_node, sub_path in sub_circuits:
            # Find the last occurrence of splice_node as a destination in path
            insert_idx = None
            for i in range(len(path) - 1, -1, -1):
                if path[i][1] == splice_node:
                    insert_idx = i + 1
                    break
            if insert_idx is None:
                # Try finding it as a source
                for i in range(len(path)):
                    if path[i][0] == splice_node:
                        insert_idx = i
                        break
            if insert_idx is not None:
                path[insert_idx:insert_idx] = sub_path
                edges_used += len(sub_path)
            else:
                # Append at end as last resort
                path.extend(sub_path)
                edges_used += len(sub_path)
        
        print(f"  [splice] Spliced {leftover_count} remaining edges into path "
              f"({len(sub_circuits)} sub-circuits). Total: {len(path)}/{total_edges}")
    
    return path


# ==============================
# 1) Graph + Basis Construction
# ==============================
np.random.seed(42)
G_directed, G_undirected = create_random_polygon_graph(12, 10, seed=100)
n, m = G_undirected.number_of_nodes(), G_undirected.number_of_edges()
nodes = sorted(G_undirected.nodes())
node_to_idx = {v: i for i, v in enumerate(nodes)}

edge_list = sorted([tuple(sorted(e)) for e in G_undirected.edges()])
edge_to_idx = {e: i for i, e in enumerate(edge_list)}

basis_meta, _ = find_horton_gram_mcb(G_undirected, edge_list)
k = len(basis_meta)

# Unoriented cycle incidence matrix C (m x k)
C_matrix = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    for i in range(m):
        if (c_data["bv"] >> i) & 1:
            C_matrix[i, j] = 1

# Oriented cycle matrix C_oriented (m x k), aligned to edge_list order
C_oriented = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    cycle_edges = [edge_list[i] for i in range(m) if C_matrix[i, j] == 1]
    cycle_subgraph = nx.Graph(cycle_edges)
    try:
        cycle_nodes = [e[0] for e in nx.find_cycle(cycle_subgraph)]
    except Exception:
        cycle_nodes = list(cycle_subgraph.nodes())

    for i in range(len(cycle_nodes)):
        u_curr, v_curr = cycle_nodes[i], cycle_nodes[(i + 1) % len(cycle_nodes)]
        u, v = sorted((u_curr, v_curr))
        if (u, v) in edge_to_idx:
            idx = edge_to_idx[(u, v)]
            C_oriented[idx, j] = +1 if (u_curr == u and v_curr == v) else -1

# Absolute node-edge incidence (n x m)
A_abs = np.zeros((n, m), dtype=int)
for j, (u, v) in enumerate(edge_list):
    A_abs[node_to_idx[u], j] = 1
    A_abs[node_to_idx[v], j] = 1


# ==============================
# 2) Parameters
# ==============================
np.random.seed(42)
r = np.round(np.random.uniform(1, 10, n), 1)
tau = np.round(np.random.uniform(1, 5, m), 1)

T_max = 800
MAX_VMIN_UPDATE_ITERS = 500
MAX_SOURCE_TRIALS = 10
PRINT_EACH_MILP_FEASIBLE_VECTORS = False
PRINT_FINAL_LATENCY_FEASIBLE_VECTORS = True

# Reward-inverse per-node latency: high reward -> low latency.
LATENCY_MIN = 80.0
LATENCY_MAX = 120.0
r_min = float(np.min(r))
r_max = float(np.max(r))
if np.isclose(r_max, r_min):
    r_norm = np.zeros_like(r)
else:
    r_norm = (r - r_min) / (r_max - r_min)
latency_requirements = np.round(
    LATENCY_MAX - r_norm * (LATENCY_MAX - LATENCY_MIN),
    1,
)

# Initial node-wise lower bounds
V_MIN_vec_initial = np.ceil(T_max / latency_requirements).astype(int)

print("\nNode reward vs latency (inverse relationship):")
for node in nodes:
    i = node_to_idx[node]
    print(
        f"  Node {node}: reward={r[i]:.1f} -> latency<={latency_requirements[i]:.1f}s, "
        f"V_MIN_i(init)={V_MIN_vec_initial[i]}"
    )

print("\nSearch node-wise feasible solution:")
print(f"  T_max={T_max}")
print("  In each iteration: solve new edge collection with current V_MIN_vec.")
print("  If node violates latency -> only that node gets V_MIN_i += 1.")

selected = None
n_vars = 2 * k + m + m
M = 20000
c_obj_B = np.zeros(n_vars)
c_obj_B[2 * k : 2 * k + m] = 0.5 * A_abs.T @ r

A1 = np.hstack([-C_oriented, C_oriented, np.eye(m), np.zeros((m, m))])
A2 = np.hstack([C_oriented, -C_oriented, np.eye(m), np.zeros((m, m))])
A3 = np.hstack([-C_oriented, C_oriented, np.eye(m), M * np.eye(m)])
A4 = np.hstack([C_oriented, -C_oriented, np.eye(m), -M * np.eye(m)])
A_time_B = np.zeros((1, n_vars))
A_time_B[0, 2 * k : 2 * k + m] = tau
A_node_B = np.zeros((n, n_vars))
A_node_B[:, 2 * k : 2 * k + m] = A_abs
A_ub_B_base = np.vstack([-A1, -A2, A3, A4, A_time_B, -A_node_B])

lb_B = np.zeros(n_vars)
ub_B = np.full(n_vars, np.inf)
ub_B[2 * k + m :] = 1

V_MIN_cap = 4 * V_MIN_vec_initial  # cap: no node exceeds 4x its initial V_MIN

# Open log file for writing iteration details
log_file = open("animate_test_log.txt", "w")
log_file.write("=" * 80 + "\n")
log_file.write("ANIMATE_TEST.PY - A* V_MIN Search Log\n")
log_file.write("=" * 80 + "\n\n")

# Run A* search over V_MIN space  
search_result = astar_vmin_search(
    V_MIN_initial=V_MIN_vec_initial,
    V_MIN_cap=V_MIN_cap,
    c_obj_B=c_obj_B,
    A_ub_B_base=A_ub_B_base,
    lb_B=lb_B,
    ub_B=ub_B,
    n_vars=n_vars,
    T_max=T_max,
    M_bigM=M,
    m=m,
    k=k,
    n=n,
    nodes=nodes,
    edge_list=edge_list,
    tau=tau,
    C_oriented=C_oriented,
    A_abs=A_abs,
    latency_requirements=latency_requirements,
    node_to_idx=node_to_idx,
    greedy_euler_fn=greedy_minmax_euler_circuit,
    max_states=500,
    max_time_sec=300,
    log_file=log_file,
    r_vec=r,
)

selected = search_result["best_feasible"]
last_successful_result = search_result["best_infeasible"]

if selected is None:
    print("\nNo latency-feasible solution found after boundary-node +1 updates.")
    if last_successful_result is not None:
        print("Using LAST SUCCESSFUL MILP result (violates latency) for animation...")
        print(f"  Violated nodes: {len(last_successful_result['violated_nodes'])}")
        print(f"  Worst gap: {last_successful_result['worst_gap']:.1f}s")
        selected = last_successful_result
    else:
        print("No successful MILP results at all. Cannot generate animation.")
        log_file.write("\nFAILED: No successful MILP results\n")
        log_file.close()
        raise SystemExit(1)

J_B = selected["J_B"]
z_opt = selected["z_opt"]
euler_path = selected["euler_path"]
edge_tau_dict = selected["edge_tau_dict"]
V_MIN_vec_final = selected["V_MIN_vec"]
latency_requirements = selected["latency_requirements"]
update_iter_final = selected["update_iter"]
print(
    "Selected solution: "
    f"update_iter={update_iter_final}, "
    f"V_MIN range=[{np.min(V_MIN_vec_final)}, {np.max(V_MIN_vec_final)}], "
    f"latency range=[{np.min(latency_requirements):.1f}, {np.max(latency_requirements):.1f}]s"
)

# Close log file
log_file.write(f"\n{'='*80}\n")
log_file.write("FINAL SOLUTION SELECTED\n")
log_file.write(f"Update iteration: {update_iter_final}\n")
log_file.write(f"V_MIN range: [{np.min(V_MIN_vec_final)}, {np.max(V_MIN_vec_final)}]\n")
log_file.write(f"Latency range: [{np.min(latency_requirements):.1f}, {np.max(latency_requirements):.1f}]s\n")
log_file.write(f"{'='*80}\n")
log_file.close()
print("Log saved to: animate_test_log.txt")


# ==============================
# 6) Animation
# ==============================
pos = nx.get_node_attributes(G_undirected, "pos") or nx.spring_layout(G_undirected, seed=42)
fig, ax = plt.subplots(figsize=(14, 10))


def update(frame):
    ax.clear()

    edge_visit_counts = {}
    cumulative_time = 0.0
    node_last_visit = {v: 0.0 for v in nodes}

    current_edge_idx = min(frame, len(euler_path) - 1)

    for idx in range(current_edge_idx + 1):
        u, v = euler_path[idx]
        key = tuple(sorted((u, v)))
        edge_visit_counts[key] = edge_visit_counts.get(key, 0) + 1

        if idx < current_edge_idx:
            cumulative_time += edge_tau_dict[(u, v)]
            node_last_visit[v] = cumulative_time

    current_time = cumulative_time

    node_urgency = {}
    for node in nodes:
        time_since = current_time - node_last_visit[node]
        node_latency = latency_requirements[node_to_idx[node]]
        urgency = min(time_since / node_latency, 1.0)
        node_urgency[node] = urgency

    node_colors = [plt.cm.RdYlGn_r(node_urgency[v]) for v in nodes]

    nx.draw_networkx_edges(G_undirected, pos, width=1, alpha=0.15, edge_color="gray", ax=ax)

    if frame > 0:
        for (u_p, v_p), count in edge_visit_counts.items():
            width = 2 + count * 0.8
            nx.draw_networkx_edges(
                G_undirected,
                pos,
                edgelist=[(u_p, v_p)],
                width=width,
                edge_color="blue",
                alpha=0.6,
                ax=ax,
            )

    if frame < len(euler_path):
        u, v = euler_path[current_edge_idx]
        nx.draw_networkx_edges(
            G_undirected,
            pos,
            [(u, v)],
            width=5,
            edge_color="red",
            ax=ax,
            alpha=0.9,
        )

    nx.draw_networkx_nodes(
        G_undirected,
        pos,
        node_color=node_colors,
        node_size=900,
        ax=ax,
        edgecolors="black",
        linewidths=2.5,
    )

    labels = {}
    for v in nodes:
        time_since = current_time - node_last_visit[v]
        node_latency = latency_requirements[node_to_idx[v]]
        labels[v] = f"{v}\\n{time_since:.1f}/{node_latency:.1f}s"
    nx.draw_networkx_labels(G_undirected, pos, labels, font_size=9, font_weight="bold", ax=ax)

    edge_labels = {
        (u, v): f"{edge_tau_dict.get((u, v), edge_tau_dict.get((v, u), 0)):.1f}s"
        for u, v in G_undirected.edges()
    }
    nx.draw_networkx_edge_labels(G_undirected, pos, edge_labels, font_size=8, ax=ax)

    if frame < len(euler_path):
        u, _ = euler_path[current_edge_idx]
        ax.plot(
            pos[u][0],
            pos[u][1],
            "ro",
            markersize=20,
            zorder=10,
            markeredgecolor="white",
            markeredgewidth=3,
        )
    elif len(euler_path) > 0:
        last_node = euler_path[-1][1]
        ax.plot(
            pos[last_node][0],
            pos[last_node][1],
            "ro",
            markersize=20,
            zorder=10,
            markeredgecolor="white",
            markeredgewidth=3,
        )

    min_vmin = int(np.min(V_MIN_vec_final))
    max_vmin = int(np.max(V_MIN_vec_final))
    min_latency = float(np.min(latency_requirements))
    max_latency = float(np.max(latency_requirements))
    ax.set_title(
        f"Method B Latency Animation (node-wise V_MIN [{min_vmin}, {max_vmin}], "
        f"latency [{min_latency:.1f}, {max_latency:.1f}]s)\\n"
        f"Reward J={J_B:.0f} | Time: {current_time:.1f}s | Step: {frame}/{len(euler_path)}",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")


frames = len(euler_path) + 10
anim = animation.FuncAnimation(fig, update, frames=frames, interval=500, repeat=True)

output_file = "visuals/A*.gif"
print(f"Saving animation to {output_file}...")
anim.save(output_file, writer="pillow", fps=2)
print(f"Saved: {output_file}")
plt.close()
