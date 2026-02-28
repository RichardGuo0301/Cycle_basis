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
        remaining.get(current).remove((best_v, best_key))
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
                try:
                    splice_node = next(n for n in comp if sub.out_degree(n) > 0)
                except StopIteration:
                    continue
            
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

os.makedirs("visuals", exist_ok=True)


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
np.random.seed(100)
r = np.round(np.random.uniform(1, 10, n), 1)
tau = np.round(np.random.uniform(1, 5, m), 1)

T_max = 800
MAX_VMIN_UPDATE_ITERS = 200
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

V_MIN_vec_work = V_MIN_vec_initial.copy()
last_selected_node = None  # Track last selected node to ensure rotation

# Open log file for writing iteration details
log_file = open("animate_random_log.txt", "w")
log_file.write("=" * 80 + "\n")
log_file.write("ANIMATE_RANDOM.PY - Iteration Log\n")
log_file.write("=" * 80 + "\n\n")

last_successful_result = None  # Track last successful MILP solve for fallback
for update_iter in range(MAX_VMIN_UPDATE_ITERS + 1):
    print(
        f"\nV_MIN update iter {update_iter}/{MAX_VMIN_UPDATE_ITERS} | "
        f"range=[{np.min(V_MIN_vec_work)}, {np.max(V_MIN_vec_work)}]"
    )

    b_ub_B = np.hstack(
        [
            np.zeros(m),
            np.zeros(m),
            M * np.ones(m),
            np.zeros(m),
            [T_max],
            -2 * V_MIN_vec_work,
        ]
    )

    result_B = milp(
        c=-c_obj_B,
        constraints=LinearConstraint(A_ub_B_base, -np.inf, b_ub_B),
        bounds=Bounds(lb=lb_B, ub=ub_B),
        integrality=np.ones(n_vars, dtype=int),
    )
    if not result_B.success:
        print(f"Optimization failed: {result_B.message}")
        print("No further +1 update is possible once MILP becomes infeasible.")
        break

    sol_B = np.round(result_B.x).astype(int)
    w_signed = sol_B[:k] - sol_B[k : 2 * k]
    z_opt = sol_B[2 * k : 2 * k + m]
    flow_B = C_oriented @ w_signed
    v_opt_B = 0.5 * A_abs @ z_opt
    J_B = r @ v_opt_B
    T_B = tau @ z_opt
    print(f"Method B solved: J={J_B:.0f}, T={T_B:.0f}, edges={np.sum(z_opt)}")
    
    # Write to log file
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"V_MIN update iter {update_iter}/{MAX_VMIN_UPDATE_ITERS}\n")
    log_file.write(f"V_MIN range: [{np.min(V_MIN_vec_work)}, {np.max(V_MIN_vec_work)}]\n")
    log_file.write(f"Method B solved: J={J_B:.0f}, T={T_B:.0f}, edges={np.sum(z_opt)}\n")
    log_file.write(f"\nw_signed (cycle weights):\n{w_signed.tolist()}\n")
    log_file.write(f"\nflow_B (edge flows):\n{flow_B.tolist()}\n")
    log_file.write(f"\nz_opt (edge visit counts):\n{z_opt.tolist()}\n")
    log_file.write(f"\nv_opt_B (node visit counts):\n{np.round(v_opt_B, 2).tolist()}\n")
    log_file.write(f"\nV_MIN_vec_work (current V_MIN):\n{V_MIN_vec_work.tolist()}\n")
    log_file.flush()  # Ensure it's written immediately
    
    # Print per-node V_MIN vs v_opt comparison
    print("  Node status (V_MIN vs v_opt):")
    for node in nodes:
        i = node_to_idx[node]
        v_min = V_MIN_vec_work[i]
        v_opt = v_opt_B[i]
        is_boundary = "←BOUNDARY" if np.isclose(v_opt, v_min) else ""
        print(f"    Node {node}: V_MIN={v_min:2d}, v_opt={v_opt:5.1f} {is_boundary}")
    
    if PRINT_EACH_MILP_FEASIBLE_VECTORS:
        print("  Feasible MILP vectors:")
        print(f"    V_MIN_vec={V_MIN_vec_work.tolist()}")
        print(f"    w_signed={w_signed.tolist()}")
        print(f"    flow_B={flow_B.tolist()}")  # <-- 添加这一行
        print(f"    z_opt={z_opt.tolist()}")
        print(f"    v_opt_B={np.round(v_opt_B, 2).tolist()}")

    DG_B = nx.MultiDiGraph()
    DG_B.add_nodes_from(nodes)
    for i, (u, v) in enumerate(edge_list):
        f = flow_B[i]
        if f > 0:
            for _ in range(f):
                DG_B.add_edge(u, v)
        elif f < 0:
            for _ in range(abs(f)):
                DG_B.add_edge(v, u)

    in_deg = dict(DG_B.in_degree())
    out_deg = dict(DG_B.out_degree())
    non_iso = [v for v in nodes if in_deg[v] + out_deg[v] > 0]
    balanced = all(in_deg[v] == out_deg[v] for v in non_iso) if non_iso else True
    is_weakly_conn = nx.is_weakly_connected(DG_B.subgraph(non_iso)) if non_iso else True
    print(f"Directed Eulerian check: balanced={balanced}, weakly_connected={is_weakly_conn}")
    if not (balanced and is_weakly_conn):
        print("Directed graph is not Eulerian under current constraints.")
        break

    edge_tau_dict = {}
    for i, (u, v) in enumerate(edge_list):
        edge_tau_dict[(u, v)] = tau[i]
        edge_tau_dict[(v, u)] = tau[i]

    possible_sources = [v for v in nodes if DG_B.out_degree(v) > 0]
    if not possible_sources:
        print("No valid Eulerian source nodes.")
        break
    
    # Prioritize node 0 as starting point if it has out-degree
    if 0 in possible_sources:
        possible_sources.remove(0)
        possible_sources.insert(0, 0)

    best_euler_path = None
    best_worst_gap = float("inf")
    best_violated_nodes = None
    best_tour_time = None

    for source in possible_sources[:MAX_SOURCE_TRIALS]:
        try:
            euler_trial = greedy_minmax_euler_circuit(
                DG_B, source, edge_tau_dict, latency_requirements, node_to_idx
            )
        except Exception as e:
            print(f"Greedy Euler failed for source {source}: {e}")
            continue

        cum_time = 0.0
        visit_times = {v: [0.0] for v in nodes}
        for (u, v) in euler_trial:
            cum_time += edge_tau_dict[(u, v)]
            visit_times[v].append(cum_time)
        T_tour = cum_time

        violated = []
        worst_gap = 0.0
        for node_idx, times in visit_times.items():
            node_latency = latency_requirements[node_to_idx[node_idx]]
            if len(times) >= 2:
                gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
                wrap_gap = T_tour - times[-1] + times[0]
                gaps.append(wrap_gap)
                max_gap = max(gaps)
            elif len(times) == 1:
                max_gap = T_tour
            else:
                max_gap = np.inf

            worst_gap = max(worst_gap, max_gap)
            if max_gap > node_latency:
                violated.append((node_idx, max_gap, node_latency))

        if worst_gap < best_worst_gap:
            best_worst_gap = worst_gap
            best_euler_path = euler_trial
            best_violated_nodes = violated
            best_tour_time = T_tour

        if not violated:
            print(f"Found latency-feasible sequence from source={source}, T_tour={T_tour:.1f}s")
            break
            
    # Write latency check results to log
    if best_violated_nodes is not None:
        log_file.write(f"Latency check: violated_nodes={len(best_violated_nodes)}, worst_gap={best_worst_gap:.1f}s\n")
        log_file.flush()

    if best_euler_path is None:
        print("Failed to generate Eulerian sequence.")
        break
    
    # Check and print the starting node
    start_node = best_euler_path[0][0] if best_euler_path else None
    if start_node is not None:
        if start_node == 0:
            print(f"✓ Eulerian path starts from node 0")
        else:
            print(f"⚠ WARNING: Eulerian path starts from node {start_node} (not node 0)")

    if not best_violated_nodes:
        print("All nodes satisfy their fixed random latency requirements.")
        if PRINT_FINAL_LATENCY_FEASIBLE_VECTORS:
            print("  Latency-feasible vectors:")
            print(f"    V_MIN_vec={V_MIN_vec_work.tolist()}")
            print(f"    w_signed={w_signed.tolist()}")
            print(f"    flow_B={flow_B.tolist()}")
            print(f"    z_opt={z_opt.tolist()}")
            print(f"    v_opt_B={np.round(v_opt_B, 2).tolist()}")
        selected = {
            "J_B": J_B,
            "z_opt": z_opt,
            "euler_path": best_euler_path,
            "edge_tau_dict": edge_tau_dict,
            "V_MIN_vec": V_MIN_vec_work.copy(),
            "update_iter": update_iter,
            "latency_requirements": latency_requirements.copy(),
            "T_tour": best_tour_time,
            "worst_gap": best_worst_gap,
        }
        break

    # Store this iteration's result as fallback: keep the one with MINIMUM worst_gap
    current_result = {
        "J_B": J_B,
        "z_opt": z_opt,
        "euler_path": best_euler_path,
        "edge_tau_dict": edge_tau_dict,
        "V_MIN_vec": V_MIN_vec_work.copy(),
        "update_iter": update_iter,
        "latency_requirements": latency_requirements.copy(),
        "T_tour": best_tour_time,
        "violated_nodes": best_violated_nodes,  # Include violation info
        "worst_gap": best_worst_gap,
    }

    if last_successful_result is None:
        last_successful_result = current_result
    else:
        # Only update if current one is BETTER (smaller worst_gap)
        if best_worst_gap < last_successful_result["worst_gap"]:
            last_successful_result = current_result
            print(f"  [fallback updated] New best worst_gap={best_worst_gap:.1f}s (iter {update_iter})")

    # New logic: find nodes where v_opt_B[i] == V_MIN_vec[i] (at boundary)
    boundary_nodes = []
    for node in nodes:
        i = node_to_idx[node]
        if np.isclose(v_opt_B[i], V_MIN_vec_work[i]):  # Tolerance for floating point
            boundary_nodes.append(node)

    if not boundary_nodes:
        print("No boundary nodes (all v_opt > V_MIN). Latency violations remain but cannot tighten further.")
        print("Stopping iteration.")
        break

    # Select ONE node to increment RANDOMLY (as requested)
    selected_node = np.random.choice(boundary_nodes)
    
    i_selected = node_to_idx[selected_node]

    print(
        f"Sequence violates latency at {len(best_violated_nodes)} nodes "
        f"(worst gap={best_worst_gap:.1f}s)."
    )
    print(f"Boundary nodes (v_opt == V_MIN): {boundary_nodes}")
    print(f"Selected node for V_MIN +1: {selected_node} (v_opt={v_opt_B[i_selected]:.1f}, V_MIN={V_MIN_vec_work[i_selected]})")

    V_MIN_vec_work[i_selected] += 1
    last_selected_node = selected_node  # Remember for next iteration

if selected is None:
    print("\nNo latency-feasible solution found after boundary-node +1 updates.")
    if last_successful_result is not None:
        print("Using BEST SUCCESSFUL MILP result (violates latency) for animation...")
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
worst_gap_final = selected.get("worst_gap", float("inf"))

print(
    "Selected solution: "
    f"update_iter={update_iter_final}, "
    f"worst_gap={worst_gap_final:.1f}s, "
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
print("Log saved to: animate_random_log.txt")


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

output_file = "visuals/random.gif"
print(f"Saving animation to {output_file}...")
anim.save(output_file, writer="pillow", fps=2)
print(f"Saved: {output_file}")
plt.close()
