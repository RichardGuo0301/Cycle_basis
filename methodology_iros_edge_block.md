# Methodology (IROS Draft): Robust Persistent Monitoring under Edge Blocking

## 1. Method Overview

We consider persistent monitoring on a connected graph where a nominal patrol is first synthesized as a closed flow, and then repaired online when critical edges become blocked.  
Our methodology is a **three-level robust pipeline**:

1. **Nominal patrol synthesis**: search a latency-feasible Eulerian circulation `flow_B` in the cycle-space parameterization.
2. **Optimization-based correction**: after edge blocking, solve a correction flow `Cm` so that blocked edges are removed while global patrol timing distortion is minimized.
3. **Execution-level realization**: convert corrected flow to an executable closed route with guided Eulerian splicing.

The cycle basis construction itself is omitted here and treated in the Preliminary section. In this section we only use the oriented cycle matrix `C^o` as a given representation.

---

## 2. Problem Formulation

Let `G=(V,E)` be the monitoring graph, `|V|=n`, `|E|=m`.  
Given oriented cycle matrix `C^o in {-1,0,1}^{m x k}` and integer cycle coefficients `w in Z^k`, any integer circulation can be represented as

`flow = C^o w`.

For each undirected edge `e`, `tau_e > 0` denotes travel time.  
For each node `v`, `L_v` denotes maximum revisit latency requirement.

We seek a nominal flow `flow_B` and route `Pi_B` such that:

- `flow_B` is Eulerian-realizable (balanced + weakly connected active subgraph),
- the induced closed walk satisfies node latency constraints,
- travel budget and node-coverage lower bounds are respected.

After disturbance, two active and non-adjacent blocked edges `e_b1, e_b2` are selected.  
Define correction flow:

`Cm = C^o beta,  beta in Z^k`.

Corrected flow:

`flow_corr = flow_B + Cm`.

Hard blocking constraints:

`flow_corr[e_b1] = 0`, `flow_corr[e_b2] = 0`.

---

## 3. Nominal Patrol Synthesis

### 3.1 MILP parameterization and frontier search

We solve a MILP over cycle coefficients and edge-activation variables to get candidate nominal flows.  
The MILP is repeatedly called inside a **BFS + priority frontier** over node-wise minimum-visit vector `V_min`, where priority is:

- first: total increments in `V_min`,
- second: larger utility score `J`.

At each frontier state:

1. Solve MILP to obtain `flow_B`.
2. Check Eulerian realizability.
3. Run urgency-aware backtracking Euler traversal.
4. Perform two-cycle latency verification.
5. If violated, expand boundary nodes in `V_min` and continue.

This produces a latency-feasible baseline `flow_B` and a corresponding nominal route `Pi_B`.

### 3.2 Urgency-aware Euler traversal

Given candidate multigraph induced by `flow_B`, traversal chooses next edge with urgency score proportional to normalized waiting time, while pruning by a shortest-path lower-bound feasibility test.  
This integrates combinatorial Eulerian sequencing with latency-awareness instead of purely post-hoc checking.

---

## 4. Disturbance Model and Correction Objectives

When two active edges are blocked, we enforce their corrected flows to zero. The correction objective is **lexicographic**:

1. Keep global tour-time change near zero (`Delta T approx 0`),
2. among such solutions, use minimum-energy correction (`min ||Cm||_2^2`).

Define:

- `T_orig = sum_e tau_e |flow_B[e]|`,
- `T_new  = sum_e tau_e |flow_corr[e]|`,
- `Delta T = T_new - T_orig`.

The practical meaning is to preserve the nominal monitoring tempo while minimizing structural disturbance to the flow.

---

## 5. Cascaded Correction Solver (Key Robustness Component)

### 5.1 Primary solver: Two-stage MIQP (from `edge_block_delta.py`)

#### Stage 1: minimize `|Delta T|`

Variables:

- `beta in Z^k`,
- `Cm = C^o beta`,
- `f = flow_B + Cm`,
- `z_abs in Z_+^m`, `b_sign in {0,1}^m` for exact `|f|` linearization,
- `eta >= 0` upper-bounding `|Delta T|`.

With Big-M linearization, enforce `z_abs[e] = |f[e]|`, then solve:

`min eta`  
s.t. blocked-edge equalities and `eta >= +/- (tau^T z_abs - T_orig)`.

#### Stage 2: minimum-norm refinement

Fix near-optimal timing by:

`eta <= eta* + epsilon`,

and solve:

`min ||Cm||_2^2`

under all Stage-1 constraints.  
This returns a correction with near-minimal timing drift and minimal quadratic correction energy.

### 5.2 MIQP no-solution fallback: One-stage L2+connectivity MIQP (reference to `edge_block_1MIQP.py`)

If Stage-1/Stage-2 is infeasible, numerically unstable, or times out, we switch to a more direct one-stage MIQP:

`min ||Cm||_2^2 + delta * sum_j z_j`

with:

- blocked-edge equalities,
- Big-M linking `-M z_j <= beta_j <= M z_j`,
- connectivity regularization on active cycle indices.

Connectivity is enforced on cycle-overlap graph using single-commodity flow from anchor set `A` (cycle indices containing blocked edges) to all active cycles.  
For the two-edge case, use union anchors `A = A(e_b1) union A(e_b2)`.

This fallback improves solver robustness when strict two-stage `Delta T` modeling is hard.

### 5.3 Structural fallback: detour re-routing

If optimized `flow_corr` is still not Eulerian-realizable, perform detour re-routing:

1. remove blocked edges from base graph,
2. for each blocked edge demand, enumerate top-`K` weighted shortest simple detours,
3. convert each detour to correction increment,
4. greedily keep detour that best preserves Eulerian feasibility.

This guarantees the pipeline can still output an executable correction even when MIQP output is structurally invalid.

---

## 6. Executable Route Realization by Guided Eulerian Splicing

A corrected flow vector is not yet a robot route. We build a directed multigraph exactly matching `flow_corr`, then generate a closed walk using **guided Fleury traversal**:

1. maintain remaining corrected edges,
2. prefer non-bridge choices,
3. prioritize the next edge that follows nominal route order `Pi_B`,
4. when nominal edge is blocked/exhausted, divert through valid correction edges,
5. continue until all corrected multiedges are consumed.

Hence, we preserve nominal patrol semantics when possible, while inserting only necessary correction segments from `Cm`.

---

## 7. Algorithmic Summary (Pseudocode)

```text
Algorithm 1: NominalFlowSynthesis(G, C^o, tau, L)
Input: graph G, oriented cycle matrix C^o, edge times tau, latency limits L
Output: nominal flow flow_B and nominal Euler route Pi_B

1: Initialize V_min from (T_max, L), frontier <- {V_min}
2: while frontier not empty do
3:     Pop state with smallest (increment cost, -J)
4:     Solve MILP under current V_min to obtain candidate flow f
5:     if f infeasible then continue
6:     if f is Eulerian-realizable then
7:         Pi <- urgency-aware backtracking Euler traversal on f
8:         if Pi satisfies two-cycle latency verification then
9:             return (f, Pi)
10:        end if
11:    end if
12:    Expand boundary nodes of V_min and push children
13: end while
14: return failure
```

```text
Algorithm 2: CascadedEdgeBlockCorrection(flow_B, blocked edges, C^o, tau)
Input: nominal flow flow_B, blocked edges {e_b1,e_b2}
Output: corrected flow flow_corr, correction Cm

1: Build constraints flow_corr[e_b1]=0 and flow_corr[e_b2]=0 with Cm=C^o beta
2: Try two-stage MIQP:
3:     Stage 1: minimize |Delta T| via exact absolute-value linearization
4:     Stage 2: minimize ||Cm||_2^2 subject to |Delta T| <= eta* + epsilon
5: if two-stage MIQP fails then
6:     Solve one-stage fallback MIQP:
7:         minimize ||Cm||_2^2 + delta * sum(z)
8:         enforce active-cycle connectivity on overlap graph from anchor set A
9: end if
10: flow_corr <- flow_B + Cm
11: if flow_corr is not Eulerian-realizable then
12:     flow_corr <- detour-rerouting fallback(flow_B, blocked edges)
13: end if
14: return (flow_corr, Cm)
```

```text
Algorithm 3: GuidedEulerSplice(flow_corr, Pi_B)
Input: corrected flow flow_corr, nominal route Pi_B
Output: executable corrected route Pi_corr

1: Build directed multigraph H exactly from flow_corr multiplicities
2: Set current node from first feasible node in Pi_B
3: while H has remaining edges do
4:     Compute valid outgoing edges (prefer non-bridge)
5:     if edge matching forward order in Pi_B exists then
6:         choose that edge
7:     else
8:         choose first valid correction edge (detour/splice)
9:     end if
10:    remove chosen edge from H and append to Pi_corr
11: end while
12: return Pi_corr
```

---

## 8. Discussion of Robustness and Practicality

The methodology is designed to avoid single-point failure:

- If strict timing-preserving MIQP is hard, fallback to simpler L2+connectivity MIQP.
- If optimizer output is not executable, fallback to graph-level detour repair.
- If corrected flow is valid but route extraction is ambiguous, guided Fleury guarantees full edge-consumption sequence on corrected multigraph.

Therefore, the framework remains operational across optimization failure, structural infeasibility, and sequencing-level mismatch.

---

## 9. Implementation Mapping (for reproducibility)

This methodology corresponds to:

- Nominal synthesis + BFS/frontier + urgency DFS: `edge_block_delta.py` (same backbone used in `edge_block_1MIQP.py`),
- Primary correction (two-stage MIQP + Delta-T control): `edge_block_delta.py`,
- MIQP no-solution fallback reference (one-stage L2+connectivity): `edge_block_1MIQP.py`,
- Structural fallback (detour) and guided Euler splicing: `edge_block_delta.py`.

This mapping keeps the paper method consistent with the current codebase and avoids introducing a new planning pipeline from scratch.

