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



import contextlib
import io
import sys
import traceback

print("\n" + "="*60)
print("STARTING 100 RANDOMIZED BATCH TRIALS (Randomizing r, tau, and blocked edges)")
print("="*60)

with open("batch_results_full.txt", "w", encoding="utf-8") as out_f:
    out_f.write("Batch testing 100 random combinations of r, tau, and blocked edges\n")
    out_f.write("Map topology is identical across trials.\n\n")

success_count = 0
for trial_seed in range(100):
    try:
        # Capture standard output for this trial
        trial_out = io.StringIO()
        with contextlib.redirect_stdout(trial_out):
            # ==============================
            # 2) Parameters
            # ==============================
            np.random.seed(trial_seed)
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
            # 4) Select TWO non-adjacent blocked edges (no shared nodes), randomly
            # ==============================
            active_indices = [i for i in range(m) if flow_B[i] != 0]
            if len(active_indices) < 2:
                raise RuntimeError("Need at least 2 active edges to block two.")
            
            rng = np.random.default_rng(seed=trial_seed)
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
            
            print("\nSolving Stage 1: Minimize global time deviation |Delta T| ...")
            beta_var = cp.Variable(shape=(k,), integer=True)
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
                eta_var >= -delta_T
            ]
            
            prob_stage1 = cp.Problem(cp.Minimize(eta_var), constraints)
            prob_stage1.solve(solver=cp.SCIP, verbose=False)
            
            if prob_stage1.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"Stage 1 failed: status={prob_stage1.status}")
            
            eta_star = float(eta_var.value) if eta_var.value is not None else 0.0
            print(f"  Stage 1 found minimal achievable |Delta T| = {eta_star:.2f}s")
            
            
            print(f"\nSolving Stage 2: Minimize ||Cm||_2^2 subject to |Delta T| <= {eta_star:.2f}s + tol ...")
            TOL = 0.2
            constraints_stage2 = constraints + [
                eta_var <= eta_star + TOL
            ]
            
            prob_stage2 = cp.Problem(
                cp.Minimize(cp.sum_squares(Cm_expr)),
                constraints_stage2
            )
            prob_stage2.solve(solver=cp.SCIP, verbose=False)
            
            if prob_stage2.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"Stage 2 failed: status={prob_stage2.status}")
            
            beta_star      = np.round(beta_var.value).astype(int)
            Cm             = C_oriented @ beta_star
            flow_corrected = flow_B + Cm
            correction_method = "Two-Stage MIQP (Min T, then min L2)"
            
            l2_euler, l2_bal, l2_wc = _check_eulerian(flow_corrected)
            print(f"[Stage 2 Result] Eulerian: {l2_euler}  (balanced={l2_bal}, weakly_connected={l2_wc})")
            print(f"[Stage 2 Result] ||Cm||_2²={int(Cm @ Cm)}   ||Cm||_1={int(np.sum(np.abs(Cm)))}")
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
                G_base = G_undirected.copy()
                for bu_r, bv_r in [blocked_edge1, blocked_edge2]:
                    if G_base.has_edge(bu_r, bv_r):
                        G_base.remove_edge(bu_r, bv_r)
                for u_e, v_e in G_base.edges():
                    idx_e = edge_to_idx.get(tuple(sorted((u_e, v_e))))
                    if idx_e is not None:
                        G_base[u_e][v_e]["weight"] = tau[idx_e]
            
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
            
            # ── Guided Fleury's Algorithm ─────────────────────────────────────────────────
            euler_path_spliced = []
            G_curr = G_corr.copy()
            
            # Find a valid starting node from the original sequence that has >0 out-degree
            curr = euler_path_original[0][0]
            seq_idx = 0
            for i, (u, v) in enumerate(euler_path_original):
                if G_curr.out_degree(u) > 0:
                    curr = u
                    seq_idx = i
                    break
                    
            splice_count = 0  # We count a "splice" every time we divert from the original sequence
            
            while G_curr.number_of_edges() > 0:
                out_edges = list(G_curr.out_edges(curr, keys=True, data=True))
                
                # 1. Determine non-bridge edges (safe choices)
                valid_choices = []
                if len(out_edges) == 1:
                    valid_choices = out_edges
                else:
                    for u, v, mkey, data in out_edges:
                        G_curr.remove_edge(u, v, key=mkey)
                        # A valid edge in an Eulerian directed graph must not disjoint the remaining graph,
                        # and MUST leave the next node `v` with > 0 out-edges (unless it's the very last edge).
                        is_last = (G_curr.number_of_edges() == 0)
                        if is_last or G_curr.out_degree(v) > 0:
                            active_nodes = [n for n in G_curr.nodes() if G_curr.degree(n) > 0]
                            if not active_nodes:
                                valid_choices.append((u, v, mkey, data))
                            else:
                                sub = G_curr.subgraph(active_nodes)
                                if nx.number_weakly_connected_components(sub) == 1:
                                    valid_choices.append((u, v, mkey, data))
                        G_curr.add_edge(u, v, key=mkey, **data)
                    
                    if not valid_choices:
                        valid_choices = out_edges  # Fallback if precision fails
                        
                # 2. Find preferred next step by scanning forward in euler_path_original
                chosen_edge = None
                diverted = True
                for i in range(seq_idx, len(euler_path_original)):
                    if euler_path_original[i][0] == curr:
                        pref_dst = euler_path_original[i][1]
                        # Is pref_dst amongst our valid_choices?
                        for e in valid_choices:
                            if e[1] == pref_dst:
                                chosen_edge = e
                                seq_idx = i + 1  # We jump our "bookmark" forward to resume tracking
                                diverted = False
                                break
                    if chosen_edge:
                        break
                        
                # 3. If the preferred edge is blocked or exhausted (Ccorrect=0), pick the first valid detour
                if not chosen_edge:
                    if not valid_choices:
                        print(f"CRITICAL ERROR at node {curr}! out_edges: {out_edges}")
                        print(f"Edges remaining in G_curr: {G_curr.number_of_edges()}")
                        print(f"Degrees of nodes with edges: {[(n, G_curr.in_degree(n), G_curr.out_degree(n)) for n in G_curr.nodes() if G_curr.degree(n)>0]}")
                        break
                    chosen_edge = valid_choices[0]
                    splice_count += 1
                    
                u, v, mkey, data = chosen_edge
                G_curr.remove_edge(u, v, key=mkey)
                euler_path_spliced.append((u, v))
                curr = v
            
            print(f"  Guided Eulerian Circuit completed. Total sequence diversions: {splice_count}")
            
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
            print(f"    Splices performed:     {splice_count}")
            print(f"    Path is continuous:    {continuous}" +
                  (f" (break at step {brk})" if not continuous else ""))
            print(f"    Returns to source {source_node}: {returns_to_source}")
            print(f"    Blocked edges in path: {blocked_in} (should be 0)")
            
            # ── Rigorous edge-count audit ─────────────────────────────────────────────────
            print(f"\n  Edge-Count Audit (spliced path vs flow_corrected):")
            print(f"  {'idx':>3}  {'edge':>10}  {'corrected':>10}  {'spliced':>8}  {'match':>6}")
            print(f"  {'-'*50}")
            
            # Count directed traversals: positive = u→v, negative = v→u
            actual_signed = np.zeros(m, dtype=int)
            for (su, sv) in euler_path_spliced:
                key = tuple(sorted((su, sv)))
                if key in edge_to_idx:
                    eidx = edge_to_idx[key]
                    u, v = edge_list[eidx]
                    if (su, sv) == (u, v):
                        actual_signed[eidx] += 1
                    else:
                        actual_signed[eidx] -= 1
            
            mismatch_count = 0
            for i in range(m):
                fc = int(flow_corrected[i])
                ac = int(actual_signed[i])
                ok = "✓" if fc == ac else "✗ MISMATCH"
                if fc != ac:
                    mismatch_count += 1
                print(f"  {i:>3}  {str(edge_list[i]):>10}  {fc:>10}  {ac:>8}  {ok}")
            
            print(f"\n  Total mismatches: {mismatch_count} / {m}")
            if mismatch_count > 0:
                print(f"  ⚠ The spliced path does NOT exactly reproduce flow_corrected.")
                print(f"    Reason: local splice preserves original flow_B traversals on non-blocked edges,")
                print(f"    while flow_corrected may need FEWER traversals (negative Cm) on some edges.")
                print(f"    The Dijkstra fallback detours also add non-Cm traversals.")
            else:
                print(f"  ✓ Perfect match: spliced path exactly reproduces flow_corrected!")
            
            
            # ==============================
            # 10) Done
            # ==============================
            

        
        # Analyze the trial output to see if there were 0 mismatches
        out_str = trial_out.getvalue()
        
        with open("batch_results_full.txt", "a", encoding="utf-8") as out_f:
            out_f.write(f"\n{'='*80}\n")
            out_f.write(f"TRIAL {trial_seed}\n")
            out_f.write(f"{'='*80}\n")
            if "Perfect match: spliced path exactly reproduces flow_corrected" in out_str:
                success_count += 1
                out_f.write(f"✓ STATUS: SUCCESS (0 mismatches)\n\n")
            else:
                out_f.write(f"✗ STATUS: MISMATCH DETECTED or OTHER ERROR\n\n")
            out_f.write(out_str)
            
        print(f"Trial {trial_seed}: Done")

    except Exception as e:
        print(f"Trial {trial_seed}: FAILED with exception {e}")
        with open("batch_results_full.txt", "a", encoding="utf-8") as out_f:
            out_f.write(f"\n{'='*80}\n")
            out_f.write(f"TRIAL {trial_seed}\n")
            out_f.write(f"{'='*80}\n")
            out_f.write(f"✗ STATUS: FAILED EXCEPTION\n")
            out_f.write(traceback.format_exc())

print(f"\nBatch Summary: {success_count} / 100 trials succeeded with PERFECT MATCH (0 mismatches).")
