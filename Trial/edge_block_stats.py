"""
Large-scale edge-blocking feasibility experiment.

For N_CONFIGS different (r, tau) parameter seeds:
  1. Solve MILP → get flow_B
  2. For each active edge in flow_B: block it → correction MILP → Eulerian check → DFS
  3. Record result (feasible / violated / not_eulerian / dfs_timeout)

Runs until total_trials >= TARGET_TRIALS (default 10000).
Saves results to edge_block_stats.csv.
"""

import numpy as np
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds
import cvxpy as cp
import math
import time
import csv
import sys
import heapq

from cycle_basis_demo import create_random_polygon_graph
from horton_gram_mcb import find_horton_gram_mcb


# ==============================
# Fixed: graph + cycle basis
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

C_matrix = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    for i in range(m):
        if (c_data["bv"] >> i) & 1:
            C_matrix[i, j] = 1

C_oriented = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    cycle_edges = [edge_list[i] for i in range(m) if C_matrix[i, j] == 1]
    csg = nx.Graph(cycle_edges)
    try:
        cycle_nodes = [e[0] for e in nx.find_cycle(csg)]
    except Exception:
        cycle_nodes = list(csg.nodes())
    for i in range(len(cycle_nodes)):
        u_c, v_c = cycle_nodes[i], cycle_nodes[(i + 1) % len(cycle_nodes)]
        u, v = sorted((u_c, v_c))
        if (u, v) in edge_to_idx:
            idx = edge_to_idx[(u, v)]
            C_oriented[idx, j] = +1 if (u_c == u and v_c == v) else -1

A_abs = np.zeros((n, m), dtype=int)
for j, (u, v) in enumerate(edge_list):
    A_abs[node_to_idx[u], j] = 1
    A_abs[node_to_idx[v], j] = 1

# Fixed MILP structural matrices (don't depend on tau or r)
M_big = 20000
A1 = np.hstack([-C_oriented, C_oriented, np.eye(m), np.zeros((m, m))])
A2 = np.hstack([C_oriented, -C_oriented, np.eye(m), np.zeros((m, m))])
A3 = np.hstack([-C_oriented, C_oriented, np.eye(m), M_big * np.eye(m)])
A4 = np.hstack([C_oriented, -C_oriented, np.eye(m), -M_big * np.eye(m)])
A_node_B = np.zeros((n, 2 * k + m + m))
A_node_B[:, 2 * k : 2 * k + m] = A_abs

lb_B = np.zeros(2 * k + m + m)
ub_B = np.full(2 * k + m + m, np.inf)
ub_B[2 * k + m:] = 1

LATENCY_MIN = 80.0
LATENCY_MAX = 200.0
T_max       = 200


# ==============================
# Helper: full V_MIN BFS+Frontier until latency-feasible flow_B
# Priority key: (total_increment, -J)
#   → explore smallest increment first (BFS completeness)
#   → break ties by largest J (Frontier reward)
# ==============================
def solve_feasible_flow(r, tau, latency_req, V_MIN_init, etd, max_iters=200):
    """
    BFS+Frontier search over V_MIN lattice.
    Returns flow_B only once DFS confirms latency-feasibility.
    Returns None if no feasible solution found within max_iters.
    """
    n_vars = 2 * k + m + m
    c_obj  = np.zeros(n_vars)
    c_obj[2 * k : 2 * k + m] = 0.5 * A_abs.T @ r
    A_time = np.zeros((1, n_vars))
    A_time[0, 2 * k : 2 * k + m] = tau
    A_ub_local = np.vstack([-A1, -A2, A3, A4, A_time, -A_node_B])

    def milp_full(V_MIN_vec):
        b_ub = np.hstack([
            np.zeros(m), np.zeros(m), M_big * np.ones(m), np.zeros(m),
            [T_max], -2 * V_MIN_vec,
        ])
        res = milp(
            c=-c_obj,
            constraints=LinearConstraint(A_ub_local, -np.inf, b_ub),
            bounds=Bounds(lb=lb_B, ub=ub_B),
            integrality=np.ones(n_vars, dtype=int),
        )
        if not res.success:
            return None
        sol      = np.round(res.x).astype(int)
        w_signed = sol[:k] - sol[k : 2 * k]
        z_opt    = sol[2 * k : 2 * k + m]
        flow_B   = C_oriented @ w_signed
        v_opt_B  = 0.5 * A_abs @ z_opt
        J        = float(r @ v_opt_B)
        return flow_B, v_opt_B, J

    frontier = []
    visited  = set()
    uid      = [0]

    def push(V_MIN_vec, cost):
        key = tuple(V_MIN_vec.tolist())
        if key in visited:
            return
        visited.add(key)
        res = milp_full(V_MIN_vec)
        if res is None:
            return
        flow_B, v_opt_B, J = res
        heapq.heappush(frontier, (cost, -J, uid[0], flow_B, v_opt_B, V_MIN_vec.copy()))
        uid[0] += 1

    push(V_MIN_init.copy(), 0)

    for _ in range(max_iters):
        if not frontier:
            break
        cost, _neg_J, _uid, flow_B, v_opt_B, V_MIN_cur = heapq.heappop(frontier)

        # Eulerian check
        DG = nx.MultiDiGraph()
        DG.add_nodes_from(nodes)
        for i, (u, v) in enumerate(edge_list):
            f = int(flow_B[i])
            if f > 0:
                for _ in range(f): DG.add_edge(u, v)
            elif f < 0:
                for _ in range(abs(f)): DG.add_edge(v, u)

        in_d = dict(DG.in_degree())
        out_d = dict(DG.out_degree())
        ni    = [v for v in nodes if in_d[v] + out_d[v] > 0]
        is_euler = (all(in_d[v] == out_d[v] for v in ni) and
                    (nx.is_weakly_connected(DG.subgraph(ni)) if ni else True))

        if is_euler:
            outcome, _, _ = run_dfs(DG, latency_req, etd, time_limit_s=DFS_TIMEOUT)
            if outcome == "feasible":
                return flow_B

        # Expand: each boundary node +1
        boundary = [nd for nd in nodes
                    if np.isclose(v_opt_B[node_to_idx[nd]], V_MIN_cur[node_to_idx[nd]])]
        if not boundary:
            continue
        for nd in boundary:
            new_V = V_MIN_cur.copy()
            new_V[node_to_idx[nd]] += 1
            push(new_V, cost + 1)

    return None


# ==============================
# Helper: correction MIQP (minimize ||Cm||_2^2 via cvxpy + SCIP)
# ==============================
def solve_correction_milp(blocked_idx, flow_B):
    """Minimize Cm^T Cm  s.t.  Cm = C_oriented @ beta,  Cm[blocked_idx] = d,  beta ∈ Z^k"""
    d        = int(-flow_B[blocked_idx])
    beta_var = cp.Variable(k, integer=True)
    Cm_expr  = C_oriented @ beta_var
    prob     = cp.Problem(cp.Minimize(cp.sum_squares(Cm_expr)),
                          [Cm_expr[blocked_idx] == d])
    prob.solve(solver=cp.SCIP, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    beta = np.round(beta_var.value).astype(int)
    return C_oriented @ beta


# ==============================
# Helper: DFS
# ==============================
def run_dfs(DG, latency_requirements, edge_tau_dict, time_limit_s=5.0):
    possible_sources = [v for v in nodes if DG.out_degree(v) > 0]
    if not possible_sources:
        return "no_source", float("inf"), 0
    if 0 in possible_sources:
        possible_sources.remove(0)
        possible_sources.insert(0, 0)

    best_worst_gap = float("inf")
    best_violated  = None

    for source in possible_sources:
        result = _backtrack(DG, source, edge_tau_dict,
                            latency_requirements, time_limit_s)
        if result is None:
            continue  # timeout/exhausted for this source

        # Simulate two cycles to correctly measure wrap-around gaps
        cum_time    = 0.0
        visit_times = {v: [] for v in nodes}
        visit_times[source].append(0.0)  # source truly visited at t=0
        for (u, v) in list(result) + list(result):
            cum_time += edge_tau_dict[(u, v)]
            visit_times[v].append(cum_time)
        T_tour = cum_time / 2.0

        violated  = []
        worst_gap = 0.0
        for nv, times in visit_times.items():
            lat = latency_requirements[node_to_idx[nv]]
            if len(times) >= 2:
                gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
                mg = max(gaps)
            else:
                mg = T_tour * 2.0
            worst_gap = max(worst_gap, mg)
            if mg > lat:
                violated.append(nv)

        if worst_gap < best_worst_gap:
            best_worst_gap = worst_gap
            best_violated  = violated

        if not violated:
            return "feasible", worst_gap, T_tour

    if best_violated is None:
        return "dfs_timeout", float("inf"), 0
    return "violated", best_worst_gap, 0


def _backtrack(DG, source, edge_tau_dict, latency_req, time_limit_s):
    start_t     = time.perf_counter()
    all_nodes   = list(DG.nodes())
    total_edges = DG.number_of_edges()

    req = {v: float(latency_req[node_to_idx[v]])
           for v in all_nodes
           if v in node_to_idx and np.isfinite(latency_req[node_to_idx[v]])}

    remaining = {u: {} for u in all_nodes}
    for u, v, _ in DG.edges(keys=True):
        remaining[u][v] = remaining[u].get(v, 0) + 1

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
        except Exception:
            dist_lb[u] = {}

    path            = []
    last_visit_time = {v: 0.0 for v in all_nodes}

    def violates(t):
        return any(t - last_visit_time[v] > r for v, r in req.items())

    def lb_impossible(cur, t):
        dmap = dist_lb.get(cur, {})
        return any((t - last_visit_time[v]) + dmap.get(v, math.inf) > r
                   for v, r in req.items())

    def dfs(cur, t):
        if time_limit_s is not None and (time.perf_counter() - start_t) > time_limit_s:
            return False
        if len(path) == total_edges:
            return cur == source and not violates(t)
        if lb_impossible(cur, t):
            return False
        neigh = remaining.get(cur, {})
        cands = [v for v, cnt in neigh.items() if cnt > 0]
        if not cands:
            return False

        def urgency(v):
            nt = t + edge_tau_dict.get((cur, v), 1.0)
            return -(nt - last_visit_time[v]) / max(req.get(v, 1.0), 1e-6)

        cands.sort(key=urgency)
        for nv in cands:
            tr  = edge_tau_dict.get((cur, nv), 1.0)
            nt  = t + tr
            path.append((cur, nv))
            neigh[nv] -= 1
            if neigh[nv] == 0:
                del neigh[nv]
            ov = last_visit_time[nv]
            last_visit_time[nv] = nt
            if not violates(nt) and dfs(nv, nt):
                return True
            path.pop()
            last_visit_time[nv] = ov
            neigh[nv] = neigh.get(nv, 0) + 1
        return False

    return path if dfs(source, 0.0) else None


# ==============================
# Main experiment loop
# ==============================
TARGET_TRIALS = 1000
DFS_TIMEOUT   = None     # seconds per DFS call (None = unlimited)

results = []  # list of dicts
total_trials  = 0
config_idx    = 0
t_global_start = time.perf_counter()

print(f"Starting large-scale experiment (target={TARGET_TRIALS} trials, DFS_timeout={DFS_TIMEOUT}s)")
print(f"Graph: n={n}, m={m}, k={k}")
print(f"{'Config':>7}  {'Edge':>4}  {'Result':>14}  {'worst_gap':>10}  {'elapsed':>8}")
print("-" * 55)

param_seed = 0
while total_trials < TARGET_TRIALS:
    param_seed += 1
    np.random.seed(param_seed)
    r   = np.round(np.random.uniform(1, 10, n), 1)
    tau = np.round(np.random.uniform(4, 8, m), 1)

    r_min, r_max = float(np.min(r)), float(np.max(r))
    if np.isclose(r_max, r_min):
        r_norm = np.zeros_like(r)
    else:
        r_norm = (r - r_min) / (r_max - r_min)
    latency_req = np.round(LATENCY_MAX - r_norm * (LATENCY_MAX - LATENCY_MIN), 1)
    V_MIN_init  = np.ceil(T_max / latency_req).astype(int)

    # Build edge_tau_dict for this config
    etd = {}
    for i, (u, v) in enumerate(edge_list):
        etd[(u, v)] = tau[i]
        etd[(v, u)] = tau[i]

    # Full V_MIN loop until latency-feasible flow_B confirmed by DFS
    flow_B = solve_feasible_flow(r, tau, latency_req, V_MIN_init, etd)
    if flow_B is None:
        continue  # no feasible solution for this config, skip

    active_edges = [i for i in range(m) if flow_B[i] != 0]
    if not active_edges:
        continue

    config_idx += 1

    for bidx in active_edges:
        if total_trials >= TARGET_TRIALS:
            break

        t0 = time.perf_counter()

        # Correction MILP
        Cm = solve_correction_milp(bidx, flow_B)
        if Cm is None:
            outcome   = "correction_failed"
            worst_gap = float("inf")
        else:
            flow_corr = flow_B + Cm

            # Eulerian check
            DG_c = nx.MultiDiGraph()
            DG_c.add_nodes_from(nodes)
            for i, (u, v) in enumerate(edge_list):
                f = int(flow_corr[i])
                if f > 0:
                    for _ in range(f): DG_c.add_edge(u, v)
                elif f < 0:
                    for _ in range(abs(f)): DG_c.add_edge(v, u)

            in_d  = dict(DG_c.in_degree())
            out_d = dict(DG_c.out_degree())
            ni    = [v for v in nodes if in_d[v] + out_d[v] > 0]
            bal   = all(in_d[v] == out_d[v] for v in ni) if ni else True
            wc    = nx.is_weakly_connected(DG_c.subgraph(ni)) if ni else True

            if not bal:
                outcome   = "not_balanced"
                worst_gap = float("inf")
            elif not wc:
                outcome   = "not_connected"
                worst_gap = float("inf")
            else:
                outcome   = "eulerian"
                worst_gap = 0.0

        elapsed = time.perf_counter() - t0
        row = {
            "param_seed":  param_seed,
            "config_idx":  config_idx,
            "blocked_idx": bidx,
            "blocked_edge": str(edge_list[bidx]),
            "flow_blocked": int(flow_B[bidx]),
            "outcome":     outcome,
            "worst_gap":   round(worst_gap, 2) if np.isfinite(worst_gap) else -1,
            "elapsed_s":   round(elapsed, 3),
        }
        results.append(row)
        total_trials += 1

        # Progress print every 100 trials
        if total_trials % 100 == 0 or total_trials <= 5:
            elapsed_total = time.perf_counter() - t_global_start
            rate = total_trials / elapsed_total
            eta  = (TARGET_TRIALS - total_trials) / rate if rate > 0 else 0
            print(f"  [{total_trials:>5}/{TARGET_TRIALS}] "
                  f"config={config_idx}, "
                  f"outcome={outcome:>14}, "
                  f"worst_gap={worst_gap:>8.1f}s, "
                  f"rate={rate:.1f} trials/s, "
                  f"ETA={eta:.0f}s")
            sys.stdout.flush()

# ==============================
# Save results
# ==============================
csv_path = "edge_block_stats.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nResults saved to {csv_path}")

# ==============================
# Statistics
# ==============================
outcomes = [r["outcome"] for r in results]
counts   = {}
for o in set(outcomes):
    counts[o] = outcomes.count(o)

total = len(results)
print(f"\n{'='*55}")
print(f"EXPERIMENT COMPLETE: {total} trials across {config_idx} parameter configs")
print(f"{'='*55}")
print(f"\nOutcome breakdown:")
for outcome, cnt in sorted(counts.items(), key=lambda x: -x[1]):
    pct = 100 * cnt / total
    print(f"  {outcome:>18}: {cnt:>6} ({pct:5.1f}%)")

eulerian_cnt    = counts.get("eulerian", 0)
not_bal_cnt     = counts.get("not_balanced", 0)
not_conn_cnt    = counts.get("not_connected", 0)
corr_fail_cnt   = counts.get("correction_failed", 0)

print(f"\nEulerian after correction: {eulerian_cnt} ({100*eulerian_cnt/total:.1f}%)")
print(f"Not balanced (should be 0): {not_bal_cnt} ({100*not_bal_cnt/total:.1f}%)")
print(f"Not connected (weak disconnection): {not_conn_cnt} ({100*not_conn_cnt/total:.1f}%)")
if corr_fail_cnt:
    print(f"Correction MILP failed: {corr_fail_cnt} ({100*corr_fail_cnt/total:.1f}%)")
print(f"\n  Not-connected cases need MILP re-optimization with hard z[j]=0 constraint")

total_time = time.perf_counter() - t_global_start
print(f"\nTotal wall time: {total_time:.1f}s  ({total_time/total:.2f}s per trial)")
