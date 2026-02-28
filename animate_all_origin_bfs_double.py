import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import milp, LinearConstraint, Bounds
import heapq
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
np.random.seed(24)
r = np.round(np.random.uniform(1, 10, n), 1)
tau = np.round(np.random.uniform(4, 8, m), 1)

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

# Build edge_tau_dict once (tau is fixed)
edge_tau_dict = {}
for i, (u, v) in enumerate(edge_list):
    edge_tau_dict[(u, v)] = tau[i]
    edge_tau_dict[(v, u)] = tau[i]


# ==============================
# Helper: solve MILP for a given V_MIN_vec
# ==============================
def solve_milp(V_MIN_vec):
    b_ub = np.hstack([
        np.zeros(m), np.zeros(m), M * np.ones(m), np.zeros(m),
        [T_max], -2 * V_MIN_vec,
    ])
    res = milp(
        c=-c_obj_B,
        constraints=LinearConstraint(A_ub_B_base, -np.inf, b_ub),
        bounds=Bounds(lb=lb_B, ub=ub_B),
        integrality=np.ones(n_vars, dtype=int),
    )
    if not res.success:
        return None
    sol = np.round(res.x).astype(int)
    w_signed = sol[:k] - sol[k:2*k]
    z_opt    = sol[2*k:2*k+m]
    flow_B   = C_oriented @ w_signed
    v_opt_B  = 0.5 * A_abs @ z_opt
    return {
        "J":        float(r @ v_opt_B),
        "T":        float(tau @ z_opt),
        "w_signed": w_signed,
        "z_opt":    z_opt,
        "flow_B":   flow_B,
        "v_opt_B":  v_opt_B,
        "V_MIN_vec": V_MIN_vec.copy(),
    }


# ==============================
# Helper: Eulerian check + DFS latency verification
# ==============================
def try_euler_dfs(milp_res):
    """Returns (euler_path, violated, T_tour) or None if not Eulerian / DFS fails."""
    flow_B = milp_res["flow_B"]
    DG = nx.MultiDiGraph()
    DG.add_nodes_from(nodes)
    for i, (u, v) in enumerate(edge_list):
        f = flow_B[i]
        if f > 0:
            for _ in range(f): DG.add_edge(u, v)
        elif f < 0:
            for _ in range(abs(f)): DG.add_edge(v, u)

    in_deg  = dict(DG.in_degree())
    out_deg = dict(DG.out_degree())
    non_iso = [v for v in nodes if in_deg[v] + out_deg[v] > 0]
    balanced = all(in_deg[v] == out_deg[v] for v in non_iso) if non_iso else True
    is_wc    = nx.is_weakly_connected(DG.subgraph(non_iso)) if non_iso else True
    if not (balanced and is_wc):
        return None

    possible_sources = [v for v in nodes if DG.out_degree(v) > 0]
    if not possible_sources:
        return None
    if 0 in possible_sources:
        possible_sources.remove(0)
        possible_sources.insert(0, 0)

    best_path = None
    best_worst_gap = float("inf")
    best_violated  = None
    best_T_tour    = None

    for source in possible_sources[:MAX_SOURCE_TRIALS]:
        trial = backtracking_euler_search(
            DG, source, edge_tau_dict, latency_requirements, node_to_idx
        )
        if trial is None:
            continue

        # Simulate TWO cycles to correctly measure wrap-around gaps
        # (avoids fake t=0 initialization for non-source nodes)
        cum_time = 0.0
        visit_times = {v: [] for v in nodes}
        visit_times[source].append(0.0)  # source starts at t=0
        double_trial = list(trial) + list(trial)  # two full cycles
        for (u, v) in double_trial:
            cum_time += edge_tau_dict[(u, v)]
            visit_times[v].append(cum_time)
        T_tour = cum_time / 2.0  # single-cycle period

        violated  = []
        worst_gap = 0.0
        for nv, times in visit_times.items():
            lat = latency_requirements[node_to_idx[nv]]
            if len(times) >= 2:
                gaps = [times[i+1]-times[i] for i in range(len(times)-1)]
                max_gap = max(gaps)
            else:
                max_gap = T_tour * 2.0
            worst_gap = max(worst_gap, max_gap)
            if max_gap > lat:
                violated.append((nv, max_gap, lat))

        if worst_gap < best_worst_gap:
            best_worst_gap = worst_gap
            best_path      = trial
            best_violated  = violated
            best_T_tour    = T_tour

        if not violated:
            print(f"    ✓ Latency-feasible! source={source}, T_tour={T_tour:.1f}s")
            break

    if best_path is None:
        return None
    return best_path, best_violated, best_T_tour


# ==============================
# BFS + Frontier search over V_MIN lattice
# Priority key: (total_increment, -J)
#   → 先找增量最小的解（BFS保完备性）
#   → 同等增量下优先 J 最大（Frontier保reward）
# ==============================
print("\n" + "=" * 60)
print("BFS + Frontier search over V_MIN lattice")
print(f"Priority: (total_increment, -J)")
print("=" * 60)

frontier      = []   # heap entries: (cost, -J, uid, milp_res)
visited_vmins = set()
uid_counter   = [0]
total_milp_calls = [0]

def push(V_MIN_vec, cost):
    key = tuple(V_MIN_vec.tolist())
    if key in visited_vmins:
        return
    visited_vmins.add(key)
    total_milp_calls[0] += 1
    res = solve_milp(V_MIN_vec)
    if res is None:
        print(f"    V_MIN={V_MIN_vec.tolist()} → MILP infeasible, pruned")
        return
    heapq.heappush(frontier, (cost, -res["J"], uid_counter[0], res))
    uid_counter[0] += 1
    print(f"    V_MIN={V_MIN_vec.tolist()} → J={res['J']:.0f}, T={res['T']:.0f}, cost={cost} → queued")

# 初始节点
push(V_MIN_vec_initial.copy(), cost=0)

selected   = None
iter_count = 0
MAX_BFS_ITERS = float("inf")

while frontier and iter_count < MAX_BFS_ITERS:
    iter_count += 1
    cost, neg_J, uid, milp_res = heapq.heappop(frontier)
    J         = -neg_J
    V_MIN_cur = milp_res["V_MIN_vec"]
    v_opt_B   = milp_res["v_opt_B"]

    print(f"\n[Iter {iter_count}] cost={cost}, J={J:.0f}, T={milp_res['T']:.0f}, "
          f"edges={np.sum(milp_res['z_opt'])}, "
          f"frontier={len(frontier)}, MILP_calls={total_milp_calls[0]}")
    print(f"  V_MIN={V_MIN_cur.tolist()}")

    # Eulerian check + DFS
    euler_result = try_euler_dfs(milp_res)

    if euler_result is not None:
        euler_path, violated, T_tour = euler_result
        if not violated:
            print(f"  *** SOLUTION FOUND! J={J:.0f}, T_tour={T_tour:.1f}s ***")
            selected = {
                "J_B":                J,
                "z_opt":              milp_res["z_opt"],
                "euler_path":         euler_path,
                "edge_tau_dict":      edge_tau_dict,
                "V_MIN_vec":          V_MIN_cur.copy(),
                "update_iter":        iter_count,
                "latency_requirements": latency_requirements.copy(),
                "T_tour":             T_tour,
                "worst_gap":          0.0,
            }
            break
        else:
            print(f"  DFS: latency violated ({len(violated)} nodes, "
                  f"worst_gap={max(g for _,g,_ in violated):.1f}s)")
    else:
        print(f"  Not Eulerian or DFS exhausted — expanding children anyway")

    # 找 boundary nodes，每个 +1 推入 frontier
    boundary = [nd for nd in nodes
                if np.isclose(v_opt_B[node_to_idx[nd]], V_MIN_cur[node_to_idx[nd]])]
    if not boundary:
        print(f"  No boundary nodes, cannot expand")
        continue

    print(f"  Boundary nodes: {boundary} → expanding {len(boundary)} children")
    for nd in boundary:
        new_V = V_MIN_cur.copy()
        new_V[node_to_idx[nd]] += 1
        push(new_V, cost + 1)

print(f"\n{'='*60}")
print(f"Search done: {iter_count} iters, {total_milp_calls[0]} MILP calls, "
      f"{len(visited_vmins)} states visited")

if selected is None:
    print("No feasible solution found.")
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
    f"Selected solution: iter={update_iter_final}, worst_gap={worst_gap_final:.1f}s, "
    f"V_MIN=[{np.min(V_MIN_vec_final)}, {np.max(V_MIN_vec_final)}], "
    f"latency=[{np.min(latency_requirements):.1f}, {np.max(latency_requirements):.1f}]s"
)


# ==============================
# 6) Animation — single-cycle & double-cycle GIFs
# ==============================
pos = nx.get_node_attributes(G_undirected, "pos") or nx.spring_layout(G_undirected, seed=42)


def make_gif(path_edges, num_cycles, output_file):
    fig, ax = plt.subplots(figsize=(14, 10))
    n_edges = len(path_edges)
    single_len = len(euler_path)

    def update(frame):
        ax.clear()
        edge_visit_counts = {}
        cumulative_time = 0.0
        node_last_visit = {v: 0.0 for v in nodes}

        current_edge_idx = min(frame, n_edges - 1)
        for idx in range(current_edge_idx + 1):
            u, v = path_edges[idx]
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
                nx.draw_networkx_edges(G_undirected, pos, edgelist=[(u_p, v_p)],
                                       width=width, edge_color="blue", alpha=0.6, ax=ax)

        if frame < n_edges:
            u, v = path_edges[current_edge_idx]
            nx.draw_networkx_edges(G_undirected, pos, [(u, v)], width=5,
                                   edge_color="red", ax=ax, alpha=0.9)

        nx.draw_networkx_nodes(G_undirected, pos, node_color=node_colors,
                               node_size=900, ax=ax, edgecolors="black", linewidths=2.5)

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

        if frame < n_edges:
            u, _ = path_edges[current_edge_idx]
            ax.plot(pos[u][0], pos[u][1], "ro", markersize=20, zorder=10,
                    markeredgecolor="white", markeredgewidth=3)
        elif n_edges > 0:
            last_node = path_edges[-1][1]
            ax.plot(pos[last_node][0], pos[last_node][1], "ro", markersize=20,
                    zorder=10, markeredgecolor="white", markeredgewidth=3)

        min_vmin = int(np.min(V_MIN_vec_final))
        max_vmin = int(np.max(V_MIN_vec_final))
        min_latency = float(np.min(latency_requirements))
        max_latency = float(np.max(latency_requirements))
        cycle_num = frame // single_len + 1 if single_len > 0 else 1
        step_in_cycle = frame % single_len if single_len > 0 else 0
        ax.set_title(
            f"Method B Latency Animation (V_MIN [{min_vmin}, {max_vmin}], "
            f"latency [{min_latency:.1f}, {max_latency:.1f}]s)\n"
            f"Reward J={J_B:.0f} | Time: {current_time:.1f}s | "
            f"Cycle {cycle_num}/{num_cycles} Step {step_in_cycle}/{single_len}",
            fontsize=14, fontweight="bold",
        )
        ax.axis("off")

    frames = n_edges + 10
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=500, repeat=True)
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer="pillow", fps=2)
    print(f"Saved: {output_file}")
    plt.close(fig)


# Single-cycle GIF
make_gif(list(euler_path), 1, "visuals/random_1cycle.gif")

# Double-cycle GIF
make_gif(list(euler_path) + list(euler_path), 2, "visuals/random_2cycle.gif")
