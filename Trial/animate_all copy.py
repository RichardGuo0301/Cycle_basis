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


def backtracking_euler_search(DG, source, edge_tau_dict, latency_req, node_to_idx, time_limit_s=None):
    """
    Exhaustive DFS backtracking search for latency-feasible Euler circuit.

    Optimizations:
      1) Symmetry compression: parallel edges (u->v) with same travel time compressed to count
      2) lb_impossible pruning: prune branches where even optimal future paths can't satisfy latency
      3) Wrap-around base case check: ensure returned paths are fully latency-feasible
      4) time_limit_s instead of max_backtracks for more stable stopping criterion

    Args:
        DG: nx.MultiDiGraph with directed edges
        source: starting node
        edge_tau_dict: dict mapping (u,v) -> travel time
        latency_req: array of per-node latency requirements
        node_to_idx: dict mapping node label -> index in latency_req
        time_limit_s: maximum wall-clock time in seconds (default 10s)

    Returns:
        list of (u,v) tuples forming complete Euler circuit, or None if timeout/infeasible
    """
    import time
    import math

    start_time = time.perf_counter()
    all_nodes = list(DG.nodes())
    total_edges = DG.number_of_edges()

    # Build requirement dict (skip missing/non-finite)
    req = {}
    for v in all_nodes:
        if v in node_to_idx:
            r = float(latency_req[node_to_idx[v]])
            if np.isfinite(r):
                req[v] = r

    # [Improvement 1] Symmetry compression: remaining[u][v] = count
    remaining = {u: {} for u in all_nodes}
    for u, v, _key in DG.edges(keys=True):
        remaining[u][v] = remaining[u].get(v, 0) + 1

    # [Improvement 2] Precompute shortest path lower bounds (Dijkstra all-pairs)
    Gw = nx.DiGraph()
    for u, v in DG.edges():
        w = float(edge_tau_dict.get((u, v), 1.0))
        if Gw.has_edge(u, v):
            Gw[u][v]["weight"] = min(Gw[u][v]["weight"], w)
        else:
            Gw.add_edge(u, v, weight=w)

    dist_lb = {}
    for u in all_nodes:
        try:
            dist_lb[u] = nx.single_source_dijkstra_path_length(Gw, u, weight="weight")
        except:
            dist_lb[u] = {}

    # State for DFS
    path = []
    last_visit_time = {v: 0.0 for v in all_nodes}
    backtrack_count = [0]

    def violates(time_now):
        """Return True if any node currently exceeds its latency bound."""
        for v, r in req.items():
            if time_now - last_visit_time[v] > r:
                return True
        return False

    def lb_impossible(current, time_now):
        """Return True if even optimal future paths cannot satisfy all latency requirements."""
        dmap = dist_lb.get(current, {})
        for v, r in req.items():
            d = dmap.get(v, math.inf)
            if (time_now - last_visit_time[v]) + d > r:
                return True
        return False

    def dfs(current, time_now):
        """
        Recursive DFS with backtracking.
        Returns True if found complete feasible path, False otherwise.
        """
        nonlocal backtrack_count

        # Check time limit
        if time_limit_s is not None and (time.perf_counter() - start_time) > time_limit_s:
            return False

        # [Improvement 3] Base case: all edges used AND wrap-around check
        if len(path) == total_edges:
            if current != source:
                return False
            # Wrap-around latency check
            for v, r in req.items():
                if time_now - last_visit_time[v] > r:
                    return False
            return True

        # [Improvement 2] lb_impossible pruning
        if lb_impossible(current, time_now):
            backtrack_count[0] += 1
            return False

        # Get remaining neighbors (count-based)
        neigh = remaining.get(current, {})
        candidates = [v for v, count in neigh.items() if count > 0]
        if not candidates:
            backtrack_count[0] += 1
            return False

        # Sort by urgency heuristic
        def urgency_score(v):
            travel_time = edge_tau_dict.get((current, v), 1.0)
            new_time = time_now + travel_time
            wait = new_time - last_visit_time[v]
            return -wait / max(req.get(v, 1.0), 1e-6)

        candidates.sort(key=urgency_score)

        # Try each candidate
        for next_v in candidates:
            travel_time = edge_tau_dict.get((current, next_v), 1.0)
            new_time = time_now + travel_time

            # Make move
            path.append((current, next_v))
            neigh[next_v] -= 1
            if neigh[next_v] == 0:
                del neigh[next_v]

            old_visit = last_visit_time[next_v]
            last_visit_time[next_v] = new_time

            # Prune if violates current latency
            if violates(new_time):
                # Undo
                path.pop()
                last_visit_time[next_v] = old_visit
                neigh[next_v] = neigh.get(next_v, 0) + 1
                backtrack_count[0] += 1
                continue

            # Recurse
            if dfs(next_v, new_time):
                return True

            # Backtrack
            path.pop()
            last_visit_time[next_v] = old_visit
            neigh[next_v] = neigh.get(next_v, 0) + 1
            backtrack_count[0] += 1

        return False

    # Start DFS
    success = dfs(source, 0.0)

    elapsed = time.perf_counter() - start_time
    if success:
        print(f"  [backtrack] Found solution with {backtrack_count[0]} backtracks in {elapsed:.2f}s")
        return path
    elif time_limit_s is not None and elapsed > time_limit_s:
        print(f"  [backtrack] TIMEOUT after {time_limit_s:.1f}s ({backtrack_count[0]} backtracks)")
        return None
    else:
        print(f"  [backtrack] EXHAUSTED search space ({backtrack_count[0]} backtracks in {elapsed:.2f}s) - no feasible path")
        return None

os.makedirs("visuals", exist_ok=True)


# ==============================
# 1) Graph + Basis Construction
# ==============================
np.random.seed(20)
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
np.random.seed(2)
r = np.round(np.random.uniform(1, 10, n), 1)
tau = np.round(np.random.uniform(1, 4, m), 1)

T_max = 200
MAX_VMIN_UPDATE_ITERS = 200
MAX_SOURCE_TRIALS = n  # Try all nodes as starting points
PRINT_EACH_MILP_FEASIBLE_VECTORS = False
PRINT_FINAL_LATENCY_FEASIBLE_VECTORS = True

# Reward-inverse per-node latency: high reward -> low latency.
LATENCY_MIN = 80.0
LATENCY_MAX = 200.0
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
    #check if it is eulerian
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

    #find possible sources and start loop
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
            euler_trial = backtracking_euler_search(
                DG_B, source, edge_tau_dict, latency_requirements, node_to_idx
            )
            
            # If backtracking returns None (timeout/exhausted), skip this source
            if euler_trial is None:
                print(f"Backtracking search failed for source {source}")
                continue
                
        except Exception as e:
            print(f"Backtracking search failed for source {source}: {e}")
            continue
        #check if the eulerian path is valid with wrap gap check
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
        print("Failed to generate Eulerian sequence (pruning blocked all paths).")
        print("  → Continuing to V_MIN+1 to give MILP more edges...")
        # Skip violation check and fallback storage, go directly to boundary+1
    else:
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

        print(
            f"Sequence violates latency at {len(best_violated_nodes)} nodes "
            f"(worst gap={best_worst_gap:.1f}s)."
        )

    # Find nodes where v_opt_B[i] == V_MIN_vec[i] (at boundary)
    boundary_nodes = []
    for node in nodes:
        i = node_to_idx[node]
        if np.isclose(v_opt_B[i], V_MIN_vec_work[i]):  # Tolerance for floating point
            boundary_nodes.append(node)

    if not boundary_nodes:
        print("No boundary nodes (all v_opt > V_MIN). Cannot tighten further.")
        print("Stopping iteration.")
        break

    # Select ONE node to increment RANDOMLY
    selected_node = np.random.choice(boundary_nodes)
    
    i_selected = node_to_idx[selected_node]

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
