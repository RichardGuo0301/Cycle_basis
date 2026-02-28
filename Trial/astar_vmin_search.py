"""
A* Best-First Search over V_MIN space.

Searches for the optimal V_MIN configuration that:
1. Keeps MILP feasible (edge flow exists)
2. Results in an Euler circuit satisfying all per-node latency requirements
3. Minimizes total edges (J) among all feasible solutions

Uses worst_gap as heuristic and MILP anti-monotonicity for pruning.
"""

import heapq
import time
import numpy as np
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds


def astar_vmin_search(
    V_MIN_initial,
    V_MIN_cap,
    c_obj_B,
    A_ub_B_base,
    lb_B,
    ub_B,
    n_vars,
    T_max,
    M_bigM,
    m,
    k,
    n,
    nodes,
    edge_list,
    tau,
    C_oriented,
    A_abs,
    latency_requirements,
    node_to_idx,
    greedy_euler_fn,
    max_states=500,
    max_time_sec=300,
    log_file=None,
    r_vec=None,
):
    """
    A* search over V_MIN configurations.

    Args:
        V_MIN_initial: initial V_MIN vector (np.array, int)
        V_MIN_cap: per-node upper bound on V_MIN (np.array, int)
        c_obj_B, A_ub_B_base, lb_B, ub_B, n_vars: MILP problem matrices
        T_max, M_bigM, m, k, n: problem dimensions
        nodes, edge_list, tau, C_oriented, A_abs: graph data
        latency_requirements: per-node latency requirements
        node_to_idx: dict mapping node -> index
        greedy_euler_fn: function(DG, source, edge_tau_dict, latency_req, node_to_idx) -> path
        max_states: max number of V_MIN configurations to evaluate
        max_time_sec: time limit in seconds
        log_file: optional file handle for logging

    Returns:
        dict with keys:
            'best_feasible': best feasible solution dict or None
            'best_infeasible': best infeasible solution (lowest worst_gap) or None
            'stats': search statistics dict
    """
    start_time = time.time()

    def _log(msg):
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    def _solve_milp(V_MIN_vec):
        """Solve MILP for given V_MIN. Returns (sol_dict, v_opt_B) or None if infeasible."""
        b_ub_B = np.hstack([
            np.zeros(m), np.zeros(m),
            M_bigM * np.ones(m), np.zeros(m),
            [T_max], -2 * V_MIN_vec,
        ])
        result = milp(
            c=-c_obj_B,
            constraints=LinearConstraint(A_ub_B_base, -np.inf, b_ub_B),
            bounds=Bounds(lb=lb_B, ub=ub_B),
            integrality=np.ones(n_vars, dtype=int),
        )
        if not result.success:
            return None

        sol = np.round(result.x).astype(int)
        w_signed = sol[:k] - sol[k : 2 * k]
        z_opt = sol[2 * k : 2 * k + m]
        flow_B = C_oriented @ w_signed
        v_opt_B = 0.5 * A_abs @ z_opt
        if r_vec is not None:
            J_B = float(np.dot(r_vec, v_opt_B))
        else:
            # Fallback if r_vec not provided (should be provided for correct J)
            J_B = float(np.sum(z_opt))

        return {
            "w_signed": w_signed,
            "z_opt": z_opt,
            "flow_B": flow_B,
            "v_opt_B": v_opt_B,
            "v_opt_B": v_opt_B,
            "J_B": J_B,  # REWARD
            "J_edges": int(np.sum(z_opt)),
            "T_tour_milp": float(tau @ z_opt),
        }

    def _build_graph_and_check(milp_sol, V_MIN_vec):
        """Build directed graph from flow, construct Euler circuit, check latency."""
        flow_B = milp_sol["flow_B"]
        z_opt = milp_sol["z_opt"]
        v_opt_B = milp_sol["v_opt_B"]

        # Build directed graph
        DG = nx.MultiDiGraph()
        DG.add_nodes_from(nodes)
        for i, (u, v) in enumerate(edge_list):
            f = flow_B[i]
            if f > 0:
                for _ in range(f):
                    DG.add_edge(u, v)
            elif f < 0:
                for _ in range(abs(f)):
                    DG.add_edge(v, u)

        # Check Eulerian property
        non_iso = [v for v in nodes if DG.in_degree(v) + DG.out_degree(v) > 0]
        if not non_iso:
            return None
        balanced = all(DG.in_degree(v) == DG.out_degree(v) for v in non_iso)
        weakly_conn = nx.is_weakly_connected(DG.subgraph(non_iso))
        if not (balanced and weakly_conn):
            return None

        # Edge tau dict
        edge_tau_dict = {}
        for i, (u, v) in enumerate(edge_list):
            edge_tau_dict[(u, v)] = tau[i]
            edge_tau_dict[(v, u)] = tau[i]

        # Try multiple sources
        possible_sources = [v for v in nodes if DG.out_degree(v) > 0]
        if 0 in possible_sources:
            possible_sources.remove(0)
            possible_sources.insert(0, 0)

        best_path = None
        best_gap = float("inf")
        best_violated = None
        best_T = None

        for source in possible_sources[:min(len(possible_sources), 10)]:
            try:
                trial = greedy_euler_fn(
                    DG, source, edge_tau_dict, latency_requirements, node_to_idx
                )
                if len(trial) != DG.number_of_edges():
                    trial = list(nx.eulerian_circuit(DG, source=source))
            except Exception:
                continue

            cum_time = 0.0
            visit_times = {v: [0.0] for v in nodes}
            for (u, v) in trial:
                cum_time += edge_tau_dict[(u, v)]
                visit_times[v].append(cum_time)
            T_tour = cum_time

            violated = []
            worst_gap = 0.0
            for nd, times in visit_times.items():
                node_lat = latency_requirements[node_to_idx[nd]]
                if len(times) >= 2:
                    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
                    gaps.append(T_tour - times[-1] + times[0])
                    max_gap = max(gaps)
                elif len(times) == 1:
                    max_gap = T_tour
                else:
                    max_gap = np.inf
                worst_gap = max(worst_gap, max_gap)
                if max_gap > node_lat:
                    violated.append((nd, max_gap, node_lat))

            if worst_gap < best_gap:
                best_gap = worst_gap
                best_path = trial
                best_violated = violated
                best_T = T_tour

            if not violated:
                break

        if best_path is None:
            return None

        # Boundary nodes
        boundary = []
        for nd in nodes:
            i = node_to_idx[nd]
            if np.isclose(v_opt_B[i], V_MIN_vec[i]):
                boundary.append(nd)

        return {
            "euler_path": best_path,
            "worst_gap": best_gap,
            "violated_nodes": best_violated,
            "T_tour": best_T,
            "boundary_nodes": boundary,
            "edge_tau_dict": edge_tau_dict,
            "edge_tau_dict": edge_tau_dict,
            "J_B": milp_sol["J_B"],
            "J_edges": milp_sol["J_edges"],
        }

    # ========== A* Search ==========
    _log("\n" + "=" * 80)
    _log("A* SEARCH over V_MIN configurations")
    _log(f"  Initial V_MIN: {V_MIN_initial.tolist()}")
    _log(f"  V_MIN cap:     {V_MIN_cap.tolist()}")
    _log(f"  Max states:    {max_states}")
    _log(f"  Time limit:    {max_time_sec}s")
    _log("=" * 80)

    # Priority queue: (worst_gap, total_vmin, state_id, V_MIN_tuple)
    # Lower worst_gap = higher priority (explored first)
    state_counter = 0
    init_tuple = tuple(V_MIN_initial.tolist())

    # First evaluate initial state
    init_milp = _solve_milp(V_MIN_initial)
    if init_milp is None:
        _log("FATAL: Initial V_MIN is already infeasible!")
        return {"best_feasible": None, "best_infeasible": None,
                "stats": {"states_explored": 0, "states_pruned": 0, "elapsed": 0}}

    init_result = _build_graph_and_check(init_milp, V_MIN_initial)
    if init_result is None:
        _log("FATAL: Initial V_MIN graph is not Eulerian!")
        return {"best_feasible": None, "best_infeasible": None,
                "stats": {"states_explored": 0, "states_pruned": 0, "elapsed": 0}}

    init_gap = init_result["worst_gap"]
    pq = [(init_gap, int(np.sum(V_MIN_initial)), state_counter, init_tuple)]
    state_counter += 1

    visited = set()
    best_feasible = None
    best_infeasible = None  # track best non-feasible for fallback
    states_explored = 0
    states_pruned_visited = 0
    states_pruned_infeasible = 0
    states_pruned_cap = 0
    feasible_count = 0

    while pq:
        # Check limits
        elapsed = time.time() - start_time
        if states_explored >= max_states:
            _log(f"\n[A*] Reached max states ({max_states}). Stopping.")
            break
        if elapsed > max_time_sec:
            _log(f"\n[A*] Time limit reached ({max_time_sec}s). Stopping.")
            break

        gap_est, vmin_sum, _, vmin_tuple = heapq.heappop(pq)

        if vmin_tuple in visited:
            states_pruned_visited += 1
            continue
        visited.add(vmin_tuple)
        states_explored += 1

        V_MIN = np.array(vmin_tuple)

        # Solve MILP
        milp_sol = _solve_milp(V_MIN)
        if milp_sol is None:
            states_pruned_infeasible += 1
            _log(f"  [A*] State #{states_explored}: V_MIN sum={vmin_sum} → MILP INFEASIBLE (pruned)")
            continue

        # Build graph and check latency
        result = _build_graph_and_check(milp_sol, V_MIN)
        if result is None:
            _log(f"  [A*] State #{states_explored}: Graph not Eulerian (skipped)")
            continue

        worst_gap = result["worst_gap"]
        n_violated = len(result["violated_nodes"])

        if n_violated == 0:
            # FEASIBLE!
            feasible_count += 1
            _log(f"\n  ★ [A*] FEASIBLE #{feasible_count} | State #{states_explored} | "
                 f"J_Reward={result['J_B']:.0f}, edges={result['J_edges']}, V_MIN sum={vmin_sum}, "
                 f"T_tour={result['T_tour']:.1f}s")
            _log(f"    V_MIN={list(vmin_tuple)}")

            sol_record = {
                "J_B": result["J_B"],
                "z_opt": milp_sol["z_opt"],
                "euler_path": result["euler_path"],
                "edge_tau_dict": result["edge_tau_dict"],
                "V_MIN_vec": V_MIN.copy(),
                "update_iter": states_explored,
                "latency_requirements": latency_requirements.copy(),
                "T_tour": result["T_tour"],
            }

            if best_feasible is None or result["J_B"] > best_feasible["J_B"]:
                best_feasible = sol_record
                _log(f"    → New best feasible! J_Reward={result['J_B']:.0f}")
            # Don't expand feasible states (adding more visits can't help)
            continue
        else:
            _log(f"  [A*] State #{states_explored}: gap={worst_gap:.1f}s, "
                 f"violated={n_violated}, J_Reward={result['J_B']:.0f}, edges={result['J_edges']}, "
                 f"V_MIN sum={vmin_sum} | queue={len(pq)}")

            # Track best infeasible for fallback
            if best_infeasible is None or worst_gap < best_infeasible["worst_gap"]:
                best_infeasible = {
                    "J_B": result["J_B"],
                    "z_opt": milp_sol["z_opt"],
                    "euler_path": result["euler_path"],
                    "edge_tau_dict": result["edge_tau_dict"],
                    "V_MIN_vec": V_MIN.copy(),
                    "update_iter": states_explored,
                    "latency_requirements": latency_requirements.copy(),
                    "T_tour": result["T_tour"],
                    "violated_nodes": result["violated_nodes"],
                    "worst_gap": worst_gap,
                }

        # Expand: try incrementing each boundary node
        boundary = result["boundary_nodes"]
        if not boundary:
            continue

        for nd in boundary:
            idx = node_to_idx[nd]
            if V_MIN[idx] >= V_MIN_cap[idx]:
                states_pruned_cap += 1
                continue
            new_vmin = list(vmin_tuple)
            new_vmin[idx] += 1
            new_tuple = tuple(new_vmin)
            if new_tuple not in visited:
                heapq.heappush(pq, (worst_gap, sum(new_vmin), state_counter, new_tuple))
                state_counter += 1

    elapsed = time.time() - start_time
    _log(f"\n{'='*80}")
    _log(f"A* SEARCH COMPLETE")
    _log(f"  States explored:         {states_explored}")
    _log(f"  States pruned (visited):  {states_pruned_visited}")
    _log(f"  States pruned (infeasible):{states_pruned_infeasible}")
    _log(f"  States pruned (cap):       {states_pruned_cap}")
    _log(f"  Feasible solutions found:  {feasible_count}")
    _log(f"  Queue remaining:           {len(pq)}")
    _log(f"  Elapsed time:              {elapsed:.1f}s")
    if best_feasible:
        _log(f"  Best feasible J_Reward:    {best_feasible['J_B']:.0f}")
        _log(f"  Best feasible V_MIN:       {best_feasible['V_MIN_vec'].tolist()}")
    if best_infeasible:
        _log(f"  Best infeasible gap:       {best_infeasible['worst_gap']:.1f}s")
    _log(f"{'='*80}")

    return {
        "best_feasible": best_feasible,
        "best_infeasible": best_infeasible,
        "stats": {
            "states_explored": states_explored,
            "states_pruned_visited": states_pruned_visited,
            "states_pruned_infeasible": states_pruned_infeasible,
            "states_pruned_cap": states_pruned_cap,
            "feasible_count": feasible_count,
            "elapsed": elapsed,
        },
    }
