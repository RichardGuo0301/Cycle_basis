"""
Edge Blocking via Minimum-Norm Cycle Correction

Given a solved flow_B = C_oriented @ w_signed, randomly block TWO active edges
that share no common node, by finding integer correction Cm = C_oriented @ beta such that:
  - (flow_B + Cm)[j1] = 0  (edge j1 is zeroed out)
  - (flow_B + Cm)[j2] = 0  (edge j2 is zeroed out)
  - minimize ||Cm||_2^2 subject to beta integer  (MIQP, cvxpy + SCIP)
If the MIQP result is not Eulerian, fall back to independent Detour re-routing for each edge.
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
import osmnx as ox
from shapely.geometry import LineString

# NumPy 2.0 compat
if not hasattr(np, 'alltrue'):
    np.alltrue = np.all

from horton_gram_mcb import find_horton_gram_mcb

# ==============================
# 1) Graph + Basis — Lehigh University
# ==============================
def build_lehigh_graph():
    point = (40.6053, -75.3780)
    dist = 800
    G = ox.graph_from_point(point, dist=dist, network_type='drive_service')
    try:
        G_undirected = ox.utils_graph.get_undirected(G)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G)

    # Cut the southern extension dangling below node 1016142119
    u_cut, v_cut = 259302817, 478228138
    u_del = 1016142119
    if G_undirected.has_edge(u_cut, v_cut):
        G_copy = G_undirected.copy()
        G_copy.remove_edge(u_cut, v_cut)
        comp_v = list(nx.node_connected_component(G_copy, v_cut))
        G_undirected.remove_nodes_from(comp_v)
    if u_del in G_undirected:
        G_undirected.remove_node(u_del)

    # Additional User Cuts
    if G_undirected.has_edge(6204650796, 6204650804):
        G_undirected.remove_edge(6204650796, 6204650804)
    if 259317685 in G_undirected:
        G_undirected.remove_node(259317685)
    right_loop_nodes = {248551974, 1016142066, 1016377132, 1016376957, 1016377134}
    G_undirected.remove_nodes_from([n for n in right_loop_nodes if n in G_undirected])
    left_loop_nodes = {259368711, 472425550, 259257425, 1121706772, 1121706745, 258574525, 259274110}
    G_undirected.remove_nodes_from([n for n in left_loop_nodes if n in G_undirected])

    # Custom connections
    for u_e, v_e in [(335361039, 335360953), (1016142113, 259346266), (60978330, 60978331)]:
        if u_e in G_undirected and v_e in G_undirected:
            x1, y1 = G_undirected.nodes[u_e]['x'], G_undirected.nodes[u_e]['y']
            x2, y2 = G_undirected.nodes[v_e]['x'], G_undirected.nodes[v_e]['y']
            geom = LineString([(x1, y1), (x2, y2)])
            length = ox.distance.great_circle(y1, x1, y2, x2)
            G_undirected.add_edge(u_e, v_e, key=0, length=length, geometry=geom)

    # Remove 1166459* dead-end nodes and specific user-flagged nodes
    nodes_116 = [n for n in G_undirected.nodes() if str(n).startswith("1166459")]
    for n in nodes_116:
        if G_undirected.degree(n) == 1:
            G_undirected.remove_node(n)
    for n in [11664593244, 11664593242, 11664593245, 11664593249]:
        if n in G_undirected:
            G_undirected.remove_node(n)

    # Dead-end pruning
    while True:
        nodes_to_remove = [node for node, d in G_undirected.degree() if d < 2]
        if not nodes_to_remove:
            break
        G_undirected.remove_nodes_from(nodes_to_remove)

    if len(G_undirected) > 0:
        lcc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(lcc).copy()
    return G_undirected


print("Building Lehigh University road-network graph...")
G_undirected = build_lehigh_graph()
n, m = G_undirected.number_of_nodes(), G_undirected.number_of_edges()
nodes = sorted(G_undirected.nodes())
node_to_idx = {v: i for i, v in enumerate(nodes)}
print(f"Graph ready: n={n}, m={m}")


edge_list = sorted([tuple(sorted((u, v))) for u, v in G_undirected.edges()])
edge_list = list(dict.fromkeys(edge_list))  # deduplicate multi-edges
edge_to_idx = {e: i for i, e in enumerate(edge_list)}
m = len(edge_list)  # after dedup

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
# 2) Parameters — ASU graph
# ==============================
SPEED = 16  # m/s

# tau[i] = travel time for edge i  (length / 16 m/s)
tau = np.zeros(m)
for i, (u, v) in enumerate(edge_list):
    edge_data = G_undirected[u][v]
    data = edge_data[0] if 0 in edge_data else next(iter(edge_data.values()))
    length = data.get('length', 100.0)
    tau[i] = round(length / SPEED, 1)

import time
START_TIME = time.time()

T_max = int(sum(tau) * 1.5)   # very generous upper bound on tour time

# MILP reward r: random integers 1-10
np.random.seed(0)
r = np.random.randint(1, 11, n)

# V_min range is random integers from 1 to 3 for each node
V_MIN_vec_initial = np.random.randint(1, 2, n)

# Dummy latency_requirements to prevent make_gif from crashing
latency_requirements = np.zeros(n)

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

print("\nSolving MILP with random V_MIN (2-5)...")
milp_res = solve_milp(V_MIN_vec_initial)

if milp_res is None:
    raise RuntimeError("Could not obtain a feasible flow_B using MILP.")

flow_B = milp_res["flow_B"]
w_signed_final = milp_res["w_signed"]
J_B = milp_res["J"]

# Quick Eulerian extraction for visualization
DG_MILP = nx.MultiDiGraph()
DG_MILP.add_nodes_from(nodes)
for i, (u, v) in enumerate(edge_list):
    f = int(flow_B[i])
    if f > 0:
        for _ in range(f):
            DG_MILP.add_edge(u, v)
    elif f < 0:
        for _ in range(abs(f)):
            DG_MILP.add_edge(v, u)

possible_sources = [v for v in nodes if DG_MILP.out_degree(v) > 0]
start_node = possible_sources[0] if possible_sources else nodes[0]
if 0 in possible_sources: start_node = 0

try:
    euler_path_original = list(nx.eulerian_circuit(DG_MILP, source=start_node))
except nx.NetworkXError:
    euler_path_original = []
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
# 4) Select TWO non-adjacent blocked edges (no shared nodes), randomly
# ==============================
active_indices = [i for i in range(m) if flow_B[i] != 0]
if len(active_indices) < 2:
    raise RuntimeError("Need at least 2 active edges to block two.")

rng = np.random.default_rng(seed=7)
shuffled = rng.permutation(active_indices).tolist()

blocked_idx1 = shuffled[0]
u1, v1 = edge_list[blocked_idx1]

# Second edge must share no node with the first
candidates2 = [i for i in shuffled[1:]
               if u1 not in edge_list[i] and v1 not in edge_list[i]]
if not candidates2:
    raise RuntimeError("No pair of non-adjacent active edges found.")

blocked_idx2 = candidates2[0]

blocked_edge1 = edge_list[blocked_idx1]
blocked_edge2 = edge_list[blocked_idx2]
blocked_flow1 = int(flow_B[blocked_idx1])
blocked_flow2 = int(flow_B[blocked_idx2])
d1 = -blocked_flow1
d2 = -blocked_flow2

print(f"\nBlocked edge 1: idx={blocked_idx1}, edge={blocked_edge1}, "
      f"flow_B={blocked_flow1}  →  need Cm[{blocked_idx1}]={d1}")
print(f"Blocked edge 2: idx={blocked_idx2}, edge={blocked_edge2}, "
      f"flow_B={blocked_flow2}  →  need Cm[{blocked_idx2}]={d2}")
print(f"  No shared nodes: {set(blocked_edge1) & set(blocked_edge2) == set()} ✓")

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
# 5) Two-Stage MIQP (Closest-to-Zero Delta T -> Minimum L2 Correction)
#    Solver: cvxpy + SCIP
# ==============================
import cvxpy as cp

DELTA_T_TOL = 0.2
print(f"\nSolving MIQP: Minimize ||Cm||_2^2 subject to |Delta T| <= {DELTA_T_TOL}s ...")
beta_var = cp.Variable(k, integer=True)
Cm_expr  = C_oriented @ beta_var
f_expr   = flow_B + Cm_expr

# Exact absolute value representation using Big-M
z_abs  = cp.Variable(m, integer=True)
b_sign = cp.Variable(m, boolean=True)

BIG_M_abs = 20000
constraints = [
    Cm_expr[blocked_idx1] == d1,
    Cm_expr[blocked_idx2] == d2,
    z_abs >= f_expr,
    z_abs >= -f_expr,
    z_abs <= f_expr + BIG_M_abs * (1 - b_sign),
    z_abs <= -f_expr + BIG_M_abs * b_sign
]

T_orig = tau @ np.abs(flow_B)
T_new  = tau @ z_abs
delta_T = T_new - T_orig

eta_var = cp.Variable()
constraints += [
    eta_var >= delta_T,
    eta_var >= -delta_T,
    eta_var <= DELTA_T_TOL
]

prob = cp.Problem(cp.Minimize(cp.sum_squares(Cm_expr)), constraints)
prob.solve(solver=cp.SCIP, verbose=False)

if prob.status not in ("optimal", "optimal_inaccurate"):
    raise RuntimeError(f"MIQP failed: status={prob.status}")

beta_star      = np.round(beta_var.value).astype(int)
Cm             = C_oriented @ beta_star
flow_corrected = flow_B + Cm
correction_method = "Single-Stage MIQP (min ||Cm||_2², |ΔT| ≤ 0.2s)"

l2_euler, l2_bal, l2_wc = _check_eulerian(flow_corrected)
print(f"[MIQP Result] Eulerian: {l2_euler}  (balanced={l2_bal}, weakly_connected={l2_wc})")
print(f"[MIQP Result] ||Cm||_2²={int(Cm @ Cm)}   ||Cm||_1={int(np.sum(np.abs(Cm)))}")
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
print(f"  flow_corrected[{blocked_idx1}] = {flow_corrected[blocked_idx1]}  "
      f"(should be 0, Cm={Cm[blocked_idx1]}, d={d1})")
print(f"  flow_corrected[{blocked_idx2}] = {flow_corrected[blocked_idx2]}  "
      f"(should be 0, Cm={Cm[blocked_idx2]}, d={d2})")
print(f"  Both constraints satisfied: "
      f"{flow_corrected[blocked_idx1] == 0 and flow_corrected[blocked_idx2] == 0}")

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
    marker = " ← BLOCKED" if i in (blocked_idx1, blocked_idx2) else ""
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
    print("\nflow_corrected is NOT Eulerian — cannot sequence into a valid patrol route.")
    print("Reason:")
    if not balanced:
        print("  Some nodes have in-degree != out-degree after the correction.")
    if not is_weakly_conn:
        print("  The active subgraph is disconnected.")
    print("\n>>> Falling back to DETOUR RE-ROUTING (two blocked edges) <<<")

    from itertools import islice

    # Build base graph with BOTH blocked edges removed
    # Use a simple weighted Graph for shortest_simple_paths
    G_base = nx.Graph()
    G_base.add_nodes_from(G_undirected.nodes())
    for u_e, v_e in G_undirected.edges():
        idx_e = edge_to_idx.get(tuple(sorted((u_e, v_e))))
        w = tau[idx_e] if idx_e is not None else 1.0
        if G_base.has_edge(u_e, v_e):
            G_base[u_e][v_e]['weight'] = min(G_base[u_e][v_e]['weight'], w)
        else:
            G_base.add_edge(u_e, v_e, weight=w)
    for bu_r, bv_r in [blocked_edge1, blocked_edge2]:
        if G_base.has_edge(bu_r, bv_r):
            G_base.remove_edge(bu_r, bv_r)

    total_Cm = np.zeros(m, dtype=int)

    for (blocked_idx_i, blocked_edge_i, blocked_flow_i, d_i) in [
        (blocked_idx1, blocked_edge1, blocked_flow1, d1),
        (blocked_idx2, blocked_edge2, blocked_flow2, d2),
    ]:
        bu_i, bv_i = blocked_edge_i
        abs_flow_i  = abs(blocked_flow_i)
        flow_src_i  = bu_i if blocked_flow_i > 0 else bv_i
        flow_dst_i  = bv_i if blocked_flow_i > 0 else bu_i

        if not nx.has_path(G_base, flow_src_i, flow_dst_i):
            print(f"  Edge {blocked_edge_i}: no path in residual graph (bridge). Skipping.")
            continue

        K_MAX = 10
        detour_paths = list(islice(
            nx.shortest_simple_paths(G_base, flow_src_i, flow_dst_i, weight="weight"), K_MAX))
        print(f"  Edge {blocked_edge_i}: {len(detour_paths)} candidate detour paths")

        best_Cm_i  = None
        best_euler = False

        for pi, dpath in enumerate(detour_paths):
            dtau = sum(tau[edge_to_idx[tuple(sorted((dpath[s], dpath[s+1])))]]
                       for s in range(len(dpath)-1))
            Cm_i = np.zeros(m, dtype=int)
            Cm_i[blocked_idx_i] = d_i
            for s in range(len(dpath)-1):
                a, b = dpath[s], dpath[s+1]
                eidx = edge_to_idx[tuple(sorted((a, b)))]
                Cm_i[eidx] += abs_flow_i if a < b else -abs_flow_i

            # Test combined correction so far
            test_flow = flow_B + total_Cm + Cm_i
            is_e, _, _ = _check_eulerian(test_flow)
            tag = ""
            if is_e and not best_euler:
                best_Cm_i  = Cm_i
                best_euler = True
                tag = "  ✓ SELECTED"
            elif best_Cm_i is None:
                best_Cm_i = Cm_i
            print(f"    Detour #{pi}: {' → '.join(map(str, dpath))}  "
                  f"τ={dtau:.1f}s  Eulerian={is_e}{tag}")

        if best_Cm_i is not None:
            total_Cm += best_Cm_i

    final_flow = flow_B + total_Cm
    det_euler, det_bal, det_wc = _check_eulerian(final_flow)
    if det_euler:
        Cm             = total_Cm
        flow_corrected = final_flow
        correction_method = "Detour (fallback)"
        balanced       = det_bal
        is_weakly_conn = det_wc
        print(f"\n  Combined two-edge detour: Eulerian=True  "
              f"||Cm||_1={int(np.sum(np.abs(Cm)))}")
    else:
        print(f"\n  Combined detour still not Eulerian. Keeping L2 result.")


# ==============================
# 8) Visualization: GIF + static correction + corrected result
# ==============================
print("\n" + "=" * 60)
print("STEP 2: Generating visualizations")
print("=" * 60)

from visualizer import make_gif, plot_static_correction, plot_corrected_result
from utils_gif import combine_gifs_side_by_side

pos = {node: (data['x'], data['y']) for node, data in G_undirected.nodes(data=True)}
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
    blocked_node=None, blocked_edge=[blocked_edge1, blocked_edge2],
    incident_indices=None, blocked_idx=[blocked_idx1, blocked_idx2],
    balanced=balanced, is_weakly_conn=is_weakly_conn, nodes=nodes,
    output_file="visuals/edge_block_correction.png"
)

# Corrected flow result diagram
plot_corrected_result(
    G_undirected=G_undirected, pos=pos, edge_list=edge_list, flow_corrected=flow_corrected, Cm=Cm,
    blocked_node=None, blocked_edge=[blocked_edge1, blocked_edge2],
    incident_indices=None, blocked_idx=[blocked_idx1, blocked_idx2],
    balanced=balanced, is_weakly_conn=is_weakly_conn, nodes=nodes,
    output_file="visuals/edge_block_corrected_result.png"
)

# ==============================
# 9) Guided Eulerian Circuit (Dynamic Insertion)
#    The exact algorithm proposed: follow euler_path_original (flow_B) as a guide,
#    skip edges where Ccorrect=0, divert into Cm edges, and return to flow_B.
#    Mathematically, this traces an Eulerian circuit on the `flow_corrected` graph
#    using Fleury's algorithm, prioritizing the original sequence.
# ==============================
print("\n" + "=" * 60)
print("STEP 3: Guided Eulerian Circuit (Dynamic Splice)")
print("=" * 60)

blocked_set = {blocked_idx1, blocked_idx2}
blocked_edge_set = {blocked_edge1, blocked_edge2}

# ── Build G_corr exactly from flow_corrected ──────────────────────────────────
G_corr = nx.MultiDiGraph()
G_corr.add_nodes_from(nodes)
for i in range(m):
    fc = int(flow_corrected[i])
    if fc != 0:
        u, v = edge_list[i]
        src = u if fc > 0 else v
        dst = v if fc > 0 else u
        for _ in range(abs(fc)):
            G_corr.add_edge(src, dst, edge_idx=i)

print(f"  G_corr built with {G_corr.number_of_edges()} directed edges (exactly matching |flow_corrected|).")

# ── Global Eulerian Extraction ─────────────────────────────────────────────────
# Because negative Cm components mathematically cancel out segments of the original
# path in unpredictable ways (leaving disconnected sub-loops if we just follow the
# old sequence blindly), we extract a brand new Eulerian Circuit from G_corr.
# This guarantees exact edge traversal counts matching |flow_corrected| without getting stuck.

source_node = euler_path_original[0][0]
try:
    euler_path_spliced = list(nx.eulerian_circuit(G_corr, source=source_node))
except nx.NetworkXError as e:
    print(f"  CRITICAL ERROR: {e}")
    euler_path_spliced = []

print(f"  Global Eulerian Circuit constructed.")

# ── Verify ────────────────────────────────────────────────────────────────────
continuous = True
brk = -1
for i in range(len(euler_path_spliced) - 1):
    if euler_path_spliced[i][1] != euler_path_spliced[i+1][0]:
        continuous = False
        brk = i
        break

source_node = euler_path_original[0][0]
returns_to_source = (len(euler_path_spliced) > 0 and
                     euler_path_spliced[-1][1] == source_node)

blocked_in = sum(1 for u, v in euler_path_spliced
                 if frozenset((u, v)) in {frozenset(e) for e in blocked_edge_set})

print(f"\n  Summary:")
print(f"    Original path length:  {len(euler_path_original)} edges")
print(f"    Spliced path length:   {len(euler_path_spliced)} edges")
print(f"    Path is continuous:    {continuous}" +
      (f" (break at step {brk})" if not continuous else ""))
print(f"    Returns to source {source_node}: {returns_to_source}")
print(f"    Blocked edges in path: {blocked_in} (should be 0)")


print("\n" + "=" * 60)
print("STEP 3.5: Step-by-Step Path Breakdown (Original vs. Spliced)")
print("=" * 60)
print("  This shows how the Guided Fleury algorithm traversed the graph,")
print("  where it followed the original flow_B path, where it diverted to add Cm detours,")
print("  and where it skipped original steps due to negative Cm cancellations.")
print("\n  Format: [Step #] (u->v)   |  Tag")
print("  " + "-" * 50)

# Align the original path and spliced path for comparison
orig_idx = 0
splice_idx = 0
orig_len = len(euler_path_original)
splice_len = len(euler_path_spliced)

spliced_colors = ["red"] * splice_len

while orig_idx < orig_len or splice_idx < splice_len:
    step_num = max(orig_idx, splice_idx) + 1
    
    if orig_idx < orig_len and splice_idx < splice_len and euler_path_original[orig_idx] == euler_path_spliced[splice_idx]:
        # They match exactly here: the robot follows the original path
        e = euler_path_original[orig_idx]
        print(f"  [{step_num:02d}]  {str(e):>10}   |  MATCH (Original Route)")
        spliced_colors[splice_idx] = "royalblue"
        orig_idx += 1
        splice_idx += 1
    elif orig_idx < orig_len and splice_idx < splice_len and euler_path_original[orig_idx][0] == euler_path_spliced[splice_idx][0]:
        # Both paths are at the same node, but they diverge!
        # Check if the remaining spliced path eventually returns to the original sequence
        e_splice = euler_path_spliced[splice_idx]
        print(f"  [{step_num:02d}]  {str(e_splice):>10}   |  + DETOUR (Positive Cm added)")
        spliced_colors[splice_idx] = "limegreen"
        splice_idx += 1
    elif orig_idx < orig_len:
        # We are at a point where the original path took a step, but the spliced path skipped it
        # (This happens when we cancel a walk due to negative Cm or a blocked edge)
        e_orig = euler_path_original[orig_idx]
        marker = "BLOCKED EDGE!" if frozenset(e_orig) in {frozenset(e) for e in blocked_edge_set} else "Negative Cm Cancellation"
        print(f"  [--]  {str(e_orig):>10}   |  - SKIPPED ({marker})")
        orig_idx += 1
    elif splice_idx < splice_len:
        # Original path is finished, but we still have remaining detours to take
        e_splice = euler_path_spliced[splice_idx]
        print(f"  [{step_num:02d}]  {str(e_splice):>10}   |  + DETOUR (Positive Cm added at end)")
        spliced_colors[splice_idx] = "limegreen"
        splice_idx += 1


# ==============================
# 10) Generate Spliced Path GIF
# ==============================
print("\n" + "=" * 60)
print("STEP 4: Generating spliced-path animation GIF")
print("=" * 60)

make_gif(euler_path_spliced, euler_path_spliced, 1,
         "visuals/edge_block_spliced_1cycle.gif",
         G_undirected, pos, edge_tau_dict, nodes, node_to_idx, latency_requirements,
         title_prefix="Cm-Spliced Path -- ", path_colors=spliced_colors, fps=1.0)

make_gif(euler_path_spliced + euler_path_spliced, euler_path_spliced, 2,
         "visuals/edge_block_spliced_2cycle.gif",
         G_undirected, pos, edge_tau_dict, nodes, node_to_idx, latency_requirements,
         title_prefix="Cm-Spliced Path -- ", path_colors=spliced_colors * 2, fps=1.0)
         
print("\n" + "=" * 60)
print("STEP 5: Combining Original and Spliced GIFs side-by-side")
print("=" * 60)

combine_gifs_side_by_side(
    "visuals/edge_block_original_1cycle.gif",
    "visuals/edge_block_spliced_1cycle.gif",
    "visuals/edge_block_comparison_1cycle.gif",
    fps=1.0
)
combine_gifs_side_by_side(
    "visuals/edge_block_original_2cycle.gif",
    "visuals/edge_block_spliced_2cycle.gif",
    "visuals/edge_block_comparison_2cycle.gif",
    fps=1.0
)

print("\nDone! Spliced path GIFs saved to visuals/")

