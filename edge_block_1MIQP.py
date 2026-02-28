"""
Edge Blocking via L2-Norm MIQP with Connectivity Constraint

Given a solved flow_B = C_oriented @ w_signed, randomly block one active edge j
by finding integer correction Cm = C_oriented @ beta such that:
  - (flow_B + Cm)[j] = 0  (edge j is zeroed out)
  - minimize  ||Cm||_2² + δ·Σz   (L2 norm + small sparsity penalty)
  - hard constraint: active cycle bases form a connected subgraph in the
    cycle-basis overlap graph, rooted at A = {j : blocked edge ∈ supp(C_j)}
    (enforced via single-commodity network flow over the overlap graph)
"""

import numpy as np
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import os
import time
import math

from cycle_basis_demo import create_random_polygon_graph
from horton_gram_mcb import find_horton_gram_mcb


# ==============================
# 1) Graph + Basis (same as animate_all.py)
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

C_matrix = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    for i in range(m):
        if (c_data["bv"] >> i) & 1:
            C_matrix[i, j] = 1

# Cycle-basis overlap graph: (j1, j2) iff they share ≥1 original graph edge
overlap_edges = [
    (j1, j2)
    for j1 in range(k)
    for j2 in range(j1 + 1, k)
    if np.any((C_matrix[:, j1] == 1) & (C_matrix[:, j2] == 1))
]
print(f"Overlap graph: k={k} cycle bases, {len(overlap_edges)} sharing pairs")

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


# ==============================
# 2) Parameters
# ==============================
np.random.seed(149)
r = np.round(np.random.uniform(1, 10, n), 1)
tau = np.round(np.random.uniform(4, 8, m), 1)

T_max = 200
LATENCY_MIN = 80.0
LATENCY_MAX = 200.0
MAX_SOURCE_TRIALS = n
r_min, r_max = float(np.min(r)), float(np.max(r))
r_norm = (r - r_min) / (r_max - r_min) if not np.isclose(r_max, r_min) else np.zeros_like(r)
latency_requirements = np.round(LATENCY_MAX - r_norm * (LATENCY_MAX - LATENCY_MIN), 1)
V_MIN_vec_initial = np.ceil(T_max / latency_requirements).astype(int)

A_abs = np.zeros((n, m), dtype=int)
for j, (u, v) in enumerate(edge_list):
    A_abs[node_to_idx[u], j] = 1
    A_abs[node_to_idx[v], j] = 1

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

edge_tau_dict = {}
for i, (u, v) in enumerate(edge_list):
    edge_tau_dict[(u, v)] = tau[i]
    edge_tau_dict[(v, u)] = tau[i]


# ==============================
# Urgency DFS (same as animate_all_origin_bfs.py)
# ==============================
def backtracking_euler_search(DG, source, edge_tau_dict, latency_req, node_to_idx, time_limit_s=None):
    start_time = time.perf_counter()
    all_nodes = list(DG.nodes())
    total_edges = DG.number_of_edges()

    req = {}
    for v in all_nodes:
        if v in node_to_idx:
            r_val = float(latency_req[node_to_idx[v]])
            if np.isfinite(r_val):
                req[v] = r_val

    remaining = {u: {} for u in all_nodes}
    for u, v, _key in DG.edges(keys=True):
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
        except:
            dist_lb[u] = {}

    path = []
    last_visit_time = {v: 0.0 for v in all_nodes}
    backtrack_count = [0]

    def violates(time_now):
        for v, r_val in req.items():
            if time_now - last_visit_time[v] > r_val:
                return True
        return False

    def lb_impossible(current, time_now):
        dmap = dist_lb.get(current, {})
        for v, r_val in req.items():
            d = dmap.get(v, math.inf)
            if (time_now - last_visit_time[v]) + d > r_val:
                return True
        return False

    def dfs(current, time_now):
        if time_limit_s is not None and (time.perf_counter() - start_time) > time_limit_s:
            return False
        if len(path) == total_edges:
            if current != source:
                return False
            for v, r_val in req.items():
                if time_now - last_visit_time[v] > r_val:
                    return False
            return True
        if lb_impossible(current, time_now):
            backtrack_count[0] += 1
            return False

        neigh = remaining.get(current, {})
        candidates = [v for v, count in neigh.items() if count > 0]
        if not candidates:
            backtrack_count[0] += 1
            return False

        def urgency_score(v):
            travel_time = edge_tau_dict.get((current, v), 1.0)
            new_time = time_now + travel_time
            wait = new_time - last_visit_time[v]
            return -wait / max(req.get(v, 1.0), 1e-6)

        candidates.sort(key=urgency_score)
        for next_v in candidates:
            travel_time = edge_tau_dict.get((current, next_v), 1.0)
            new_time = time_now + travel_time

            path.append((current, next_v))
            neigh[next_v] -= 1
            if neigh[next_v] == 0:
                del neigh[next_v]
            old_visit = last_visit_time[next_v]
            last_visit_time[next_v] = new_time

            if violates(new_time):
                path.pop()
                last_visit_time[next_v] = old_visit
                neigh[next_v] = neigh.get(next_v, 0) + 1
                backtrack_count[0] += 1
                continue

            if dfs(next_v, new_time):
                return True

            path.pop()
            last_visit_time[next_v] = old_visit
            neigh[next_v] = neigh.get(next_v, 0) + 1
            backtrack_count[0] += 1

            if current == source and len(path) == 0:
                return False

        return False

    success = dfs(source, 0.0)
    elapsed = time.perf_counter() - start_time
    if success:
        print(f"  [backtrack] Found solution with {backtrack_count[0]} backtracks in {elapsed:.2f}s")
        return path
    elif time_limit_s is not None and elapsed > time_limit_s:
        print(f"  [backtrack] TIMEOUT after {time_limit_s:.1f}s ({backtrack_count[0]} backtracks)")
        return None
    else:
        print(f"  [backtrack] EXHAUSTED search space ({backtrack_count[0]} backtracks in {elapsed:.2f}s)")
        return None


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
# Helper: Eulerian check + urgency DFS + two-cycle verification
# ==============================
def try_euler_dfs(milp_res):
    """Returns (euler_path, violated, T_tour) or None."""
    flow_B_local = milp_res["flow_B"]
    DG = nx.MultiDiGraph()
    DG.add_nodes_from(nodes)
    for i, (u, v) in enumerate(edge_list):
        f = flow_B_local[i]
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

        # Two-cycle verification
        cum_time = 0.0
        visit_times = {v: [] for v in nodes}
        visit_times[source].append(0.0)
        double_trial = list(trial) + list(trial)
        for (u, v) in double_trial:
            cum_time += edge_tau_dict[(u, v)]
            visit_times[v].append(cum_time)
        T_tour = cum_time / 2.0

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
# ==============================
print("BFS + Frontier search over V_MIN lattice")
print(f"Priority: (total_increment, -J)")
print("=" * 60)

frontier      = []
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

push(V_MIN_vec_initial.copy(), cost=0)

selected   = None
iter_count = 0
flow_B              = None
w_signed_final      = None
euler_path_original = None
J_B                 = None

while frontier:
    iter_count += 1
    cost, neg_J, uid, milp_res = heapq.heappop(frontier)
    J         = -neg_J
    V_MIN_cur = milp_res["V_MIN_vec"]
    v_opt_B   = milp_res["v_opt_B"]

    print(f"\n[Iter {iter_count}] cost={cost}, J={J:.0f}, T={milp_res['T']:.0f}, "
          f"edges={np.sum(milp_res['z_opt'])}, "
          f"frontier={len(frontier)}, MILP_calls={total_milp_calls[0]}")
    print(f"  V_MIN={V_MIN_cur.tolist()}")

    euler_result = try_euler_dfs(milp_res)

    if euler_result is not None:
        euler_path, violated, T_tour = euler_result
        if not violated:
            print(f"  *** SOLUTION FOUND! J={J:.0f}, T_tour={T_tour:.1f}s ***")
            flow_B              = milp_res["flow_B"]
            w_signed_final      = milp_res["w_signed"]
            euler_path_original = euler_path
            J_B                 = J
            break
        else:
            print(f"  DFS: latency violated ({len(violated)} nodes, "
                  f"worst_gap={max(g for _,g,_ in violated):.1f}s)")
    else:
        print(f"  Not Eulerian or DFS exhausted — expanding children anyway")

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

if flow_B is None:
    raise RuntimeError("Could not obtain a latency-feasible flow_B.")

print(f"\nOriginal flow_B is latency-feasible ✓")
print(f"  flow_B = {flow_B.tolist()}")


# ==============================
# 3) Print original flow_B
# ==============================
print("\n" + "=" * 60)
print("Original flow_B (= C_oriented @ w_signed):")
print(f"  {flow_B.tolist()}")
print(f"  Non-zero edges: {[(edge_list[i], flow_B[i]) for i in range(m) if flow_B[i] != 0]}")


# ==============================
# 4) Select blocked edge j (first active edge)
# ==============================
active_indices = [i for i in range(m) if flow_B[i] != 0]
if not active_indices:
    raise RuntimeError("flow_B is all zeros, nothing to block.")

blocked_idx = active_indices[0]  # deterministically pick first active edge
blocked_edge = edge_list[blocked_idx]
blocked_flow = int(flow_B[blocked_idx])
d = -blocked_flow  # required Cm[blocked_idx]

print(f"\nBlocked edge index: {blocked_idx} (active_indices[0])")
print(f"  Edge: {blocked_edge}")
print(f"  flow_B[{blocked_idx}] = {blocked_flow}  →  need Cm[{blocked_idx}] = {d}")


# ── Eulerian quick check helper (used in Section 5 and STEP 1) ───────────────
def _check_eulerian(flow_vec):
    DG_tmp = nx.MultiDiGraph()
    DG_tmp.add_nodes_from(nodes)
    for i, (u, v) in enumerate(edge_list):
        f = int(flow_vec[i])
        if f > 0:
            for _ in range(f): DG_tmp.add_edge(u, v)
        elif f < 0:
            for _ in range(abs(f)): DG_tmp.add_edge(v, u)
    in_d  = dict(DG_tmp.in_degree())
    out_d = dict(DG_tmp.out_degree())
    ni    = [v for v in nodes if in_d[v] + out_d[v] > 0]
    bal   = all(in_d[v] == out_d[v] for v in ni) if ni else True
    wc    = nx.is_weakly_connected(DG_tmp.subgraph(ni)) if ni else True
    return bal and wc, bal, wc


# ==============================
# 5) MIQP: minimize ||Cm||_2² + δ·Σz
#    s.t. Cm[blocked_idx] = d,  Cm = C_oriented @ beta,  beta ∈ Z^k
#    Connectivity: active cycle bases reachable from A in overlap graph
#      via single-commodity flow with virtual supersource → A nodes
# ==============================
print("\nSolving MIQP (L2 + connectivity):  min ||Cm||_2² + δ·Σz ...")

# A = cycle bases whose support contains the blocked edge
A_set = [j for j in range(k) if C_matrix[blocked_idx, j] == 1]
A_idx  = {j: i for i, j in enumerate(A_set)}   # j → position in A_set
print(f"  A (bases containing blocked edge {blocked_edge}): {A_set}")

M_bigM = 20      # big-M for beta ↔ z linking  (|beta[j]| ≤ 20 is always sufficient)
DELTA  = 1e-3    # tiny sparsity penalty to prevent spurious z[j]=1 when beta[j]=0

beta_var = cp.Variable(k, integer=True)
z_var    = cp.Variable(k, boolean=True)
Cm_expr  = C_oriented @ beta_var

# Directed overlap arcs (both directions of each undirected sharing pair)
dir_arcs = [(j1, j2) for (j1, j2) in overlap_edges] + \
           [(j2, j1) for (j1, j2) in overlap_edges]
arc_idx  = {arc: i for i, arc in enumerate(dir_arcs)}

# Flow variables: virtual source → A nodes, and arc flows over overlap graph
f_s   = cp.Variable(len(A_set), nonneg=True)   # source → A_set[i]
f_arc = cp.Variable(len(dir_arcs), nonneg=True) # arc flows

constr = [
    # Core blocking constraint
    Cm_expr[blocked_idx] == d,
    # Big-M linking: z[j] = 0  →  beta[j] = 0
    beta_var <=  M_bigM * z_var,
    beta_var >= -M_bigM * z_var,
    # Source capacity: only flow to active A nodes
    *[f_s[i] <= (k - 1) * z_var[j] for i, j in enumerate(A_set)],
    # Arc capacity: only between active-active node pairs
    *[f_arc[arc_idx[(j1, j2)]] <= (k - 1) * z_var[j1] for (j1, j2) in dir_arcs],
    *[f_arc[arc_idx[(j1, j2)]] <= (k - 1) * z_var[j2] for (j1, j2) in dir_arcs],
]

# Absorption: each active node j must absorb net ≥ 1 unit of flow
# (forces every z[j]=1 node to be reachable from the virtual source via A)
for j in range(k):
    in_terms  = ([f_s[A_idx[j]]] if j in A_idx else [])
    out_terms = []
    for (j1, j2) in dir_arcs:
        if j2 == j:
            in_terms.append(f_arc[arc_idx[(j1, j2)]])
        if j1 == j:
            out_terms.append(f_arc[arc_idx[(j1, j2)]])
    in_expr  = cp.sum(in_terms)  if in_terms  else cp.Constant(0.0)
    out_expr = cp.sum(out_terms) if out_terms else cp.Constant(0.0)
    constr.append(in_expr - out_expr >= z_var[j])

prob_L2 = cp.Problem(
    cp.Minimize(cp.sum_squares(Cm_expr) + DELTA * cp.sum(z_var)),
    constr
)
prob_L2.solve(solver=cp.SCIP, verbose=False)

if prob_L2.status not in ("optimal", "optimal_inaccurate"):
    raise RuntimeError(f"MIQP failed: status={prob_L2.status}")

beta_star      = np.round(beta_var.value).astype(int)
Cm             = C_oriented @ beta_star
flow_corrected = flow_B + Cm
correction_method = "L2+connectivity"

active_bases = [j for j in range(k) if beta_star[j] != 0]
print(f"  Active cycle bases (beta≠0): {active_bases}")

l2_euler, l2_bal, l2_wc = _check_eulerian(flow_corrected)
print(f"[L2+conn] Eulerian: {l2_euler}  (balanced={l2_bal}, weakly_connected={l2_wc})")
print(f"[L2+conn] ||Cm||_2²={int(Cm @ Cm)}   ||Cm||_1={int(np.sum(np.abs(Cm)))}")
print(f"\nFinal correction method used: {correction_method}")



# ==============================
# 6) Results
# ==============================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nbeta* (cycle basis weights for correction):")
print(f"  {beta_star.tolist()}")

print(f"\nCm = C_oriented @ beta*:")
print(f"  {Cm.tolist()}")

print(f"\nOriginal flow_B:")
print(f"  {flow_B.tolist()}")

print(f"\nflow_B + Cm (corrected flow):")
print(f"  {flow_corrected.tolist()}")

print(f"\nVerification:")
print(f"  flow_corrected[{blocked_idx}] = {flow_corrected[blocked_idx]}  (should be 0)")
print(f"  Cm[{blocked_idx}] = {Cm[blocked_idx]}  (should be {d})")
print(f"  Constraint satisfied: {flow_corrected[blocked_idx] == 0}")

print(f"\nNorm comparison:")
print(f"  ||Cm||_2² = {int(Cm @ Cm)}  (minimized by MIQP)")
print(f"  ||Cm||_1  = {int(np.sum(np.abs(Cm)))}")
print(f"  ||flow_B||_2² = {int(flow_B @ flow_B)}")
print(f"  ||flow_corrected||_2² = {int(flow_corrected @ flow_corrected)}")

delta_time = (np.abs(flow_corrected) - np.abs(flow_B)) * tau
print(f"\nTime difference (|corrected| - |flow_B|) * tau:")
print(f"  Total: {np.sum(delta_time):+.1f}s  (positive = more time, negative = less time)")

print(f"\nEdge-level comparison:")
print(f"  {'idx':>3}  {'edge':>10}  {'flow_B':>8}  {'Cm':>6}  {'corrected':>10}  {'tau':>6}  {'delta_t':>8}")
print(f"  {'-'*65}")
for i in range(m):
    marker = " ← BLOCKED" if i == blocked_idx else ""
    dt = delta_time[i]
    dt_str = f"{dt:+.1f}s" if dt != 0 else ""
    print(f"  {i:>3}  {str(edge_list[i]):>10}  {flow_B[i]:>8}  {Cm[i]:>6}  {flow_corrected[i]:>10}  {tau[i]:>5.1f}  {dt_str:>8}{marker}")


# ==============================
# 7) Step 1: Validate Eulerian property of flow_corrected
# ==============================
print("\n" + "=" * 60)
print("STEP 1: Eulerian Check on flow_corrected")
print("=" * 60)

DG_corrected = nx.MultiDiGraph()
DG_corrected.add_nodes_from(nodes)
for i, (u, v) in enumerate(edge_list):
    f = int(flow_corrected[i])
    if f > 0:
        for _ in range(f):
            DG_corrected.add_edge(u, v)
    elif f < 0:
        for _ in range(abs(f)):
            DG_corrected.add_edge(v, u)

in_deg  = dict(DG_corrected.in_degree())
out_deg = dict(DG_corrected.out_degree())
non_iso = [v for v in nodes if in_deg[v] + out_deg[v] > 0]

balanced     = all(in_deg[v] == out_deg[v] for v in non_iso) if non_iso else True
is_weakly_conn = nx.is_weakly_connected(DG_corrected.subgraph(non_iso)) if non_iso else True

print(f"\nDegree balance (in==out for all nodes): {balanced}")
for v in nodes:
    status = "OK" if in_deg[v] == out_deg[v] else "IMBALANCED"
    if in_deg[v] + out_deg[v] > 0:
        print(f"  Node {v}: in={in_deg[v]}, out={out_deg[v]}  {status}")

print(f"\nWeakly connected (non-isolated subgraph): {is_weakly_conn}")
print(f"\nEulerian circuit exists: {balanced and is_weakly_conn}")

if not (balanced and is_weakly_conn):
    print("\nflow_corrected is NOT Eulerian — connectivity constraint may need tightening.")
    print("Reason:")
    if not balanced:
        print("  Some nodes have in-degree != out-degree after the correction.")
    if not is_weakly_conn:
        print("  The active subgraph is disconnected.")


# ==============================
# 8) Visualization: GIF + static correction + corrected result
# ==============================
print("\n" + "=" * 60)
print("STEP 2: Generating visualizations")
print("=" * 60)

from visualizer import make_gif, plot_static_correction, plot_corrected_result

pos = nx.get_node_attributes(G_undirected, "pos") or nx.spring_layout(G_undirected, seed=42)
os.makedirs("visuals", exist_ok=True)

# Single-cycle GIF of original path
make_gif(list(euler_path_original), list(euler_path_original), 1,
         "visuals/edge_block_original_1cycle.gif",
         G_undirected, pos, edge_tau_dict, nodes, node_to_idx, latency_requirements,
         title_prefix="Original flow_B — ")

# Double-cycle GIF of original path
make_gif(list(euler_path_original) + list(euler_path_original), list(euler_path_original), 2,
         "visuals/edge_block_original_2cycle.gif",
         G_undirected, pos, edge_tau_dict, nodes, node_to_idx, latency_requirements,
         title_prefix="Original flow_B — ")

# Static correction diagram
plot_static_correction(
    G_undirected=G_undirected, pos=pos, edge_list=edge_list, flow_B=flow_B, Cm=Cm,
    blocked_node=None, blocked_edge=blocked_edge, incident_indices=None, blocked_idx=blocked_idx,
    balanced=balanced, is_weakly_conn=is_weakly_conn, nodes=nodes,
    output_file="visuals/edge_block_correction.png"
)

# Corrected flow result diagram
plot_corrected_result(
    G_undirected=G_undirected, pos=pos, edge_list=edge_list, flow_corrected=flow_corrected, Cm=Cm,
    blocked_node=None, blocked_edge=blocked_edge, incident_indices=None, blocked_idx=blocked_idx,
    balanced=balanced, is_weakly_conn=is_weakly_conn, nodes=nodes,
    output_file="visuals/edge_block_corrected_result.png"
)


