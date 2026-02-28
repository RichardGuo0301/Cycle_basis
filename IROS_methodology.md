# Methodology: Cycle-Space Persistent Monitoring with Online Edge-Failure Recovery

---

## III. METHODOLOGY

This section presents the complete algorithmic framework in the order it executes. We first describe the offline phase: constructing the **latency-feasible baseline patrol flow** $f_B$ via a cycle-space MILP searched over a V_MIN lattice (Section III-A through III-C). We then describe the online phase: recovering from one or two simultaneously blocked edges using an algebraic cycle-space correction followed, when necessary, by a topological Detour fallback (Section III-D through III-H). Finally, we describe how the corrected flow is executed as a concrete robot path via Guided Fleury's algorithm (Section III-I).

---

## III-A. Graph Setup and Cycle-Space Parameterization

Let $G = (V, E)$ be an undirected connected graph with $n = |V|$ nodes and $m = |E|$ edges. Each edge $e_i$ carries a travel time $\tau_i > 0$ and each node $v$ carries a scalar reward $r_v > 0$ and a latency requirement $L_v > 0$ (maximum allowable revisit interval). We fix a canonical orientation on $E$: for each edge $(u, v) \in E$ we require $u < v$ under the node ordering, yielding the oriented incidence matrix $B_{\text{oriented}} \in \{-1, 0, +1\}^{n \times m}$.

Given the Minimum Cycle Basis $\{c_1, \ldots, c_k\}$ computed by the Horton-Gram algorithm (described in Section II), we form the **signed cycle-edge incidence matrix** $\mathbf{C}_{\text{oriented}} \in \{-1, 0, +1\}^{m \times k}$ where $\mathbf{C}_{\text{oriented}}[i, j] = +1$ (resp. $-1$) if basis cycle $j$ traverses edge $i$ in the canonical (resp. reverse) direction, and $0$ if edge $i$ is not in cycle $j$. The cycle rank is $k = m - n + 1$.

A directed patrol flow is an integer vector $f \in \mathbb{Z}^m$, where $f[i] > 0$ means the robot traverses edge $i$ in the canonical direction $f[i]$ times per cycle, and $f[i] < 0$ means $|f[i]|$ traversals in the reverse direction. An Eulerian patrol flow must satisfy:

$$B_{\text{oriented}} \cdot f = \mathbf{0} \quad \text{(degree balance at every node)}$$

and the support $\{e : f[e] \neq 0\}$ must induce a weakly connected subgraph. By the cycle-space theorem, any integer flow satisfying degree balance can be written as $f = \mathbf{C}_{\text{oriented}} \cdot w$ for some $w \in \mathbb{Z}^k$. We work entirely in the $k$-dimensional **cycle-space coordinate** $w$ rather than the $m$-dimensional edge-space $f$.

---

## III-B. Latency Requirements and the V_MIN Vector

Node rewards $r_v$ are normalized to $[0, 1]$ and used to assign heterogeneous latency requirements: nodes with higher reward are visited more frequently, i.e., they have tighter latency bounds. Specifically:

$$L_v = L_{\max} - \tilde{r}_v \cdot (L_{\max} - L_{\min}), \qquad \tilde{r}_v = \frac{r_v - r_{\min}}{r_{\max} - r_{\min}}$$

where $L_{\min} = 80$ s and $L_{\max} = 200$ s in our experiments. The minimum visit count per node per cycle is:

$$V_{\min}[v] = \left\lceil \frac{T_{\max}}{L_v} \right\rceil$$

This initial $V_{\min}$ vector serves as the starting point of the BFS+Frontier lattice search described in Section III-C.

---

## III-C. Baseline Flow via MILP and BFS+Frontier Lattice Search

### III-C.1. MILP Formulation for a Fixed V_MIN

For a given visit-count vector $V_{\min} \in \mathbb{Z}^n_{\geq 0}$, we solve an Integer Linear Program to find the best cycle-space flow. The variables are:

$$x = [w^+;\ w^-;\ z;\ b] \in \mathbb{Z}_{\geq 0}^{2k+m+m}$$

where $w = w^+ - w^-$ is the signed cycle-basis weight vector split into non-negative parts, $z \in \mathbb{Z}_{\geq 0}^m$ represents the **absolute traversal count** $|f[i]|$ per edge, and $b \in \{0,1\}^m$ are Big-M sign indicators. The flow in edge space is $f = \mathbf{C}_{\text{oriented}}(w^+ - w^-)$.

**Objective — maximize total reward collected per cycle:**

$$\max_{x} \quad \frac{1}{2} \left( A_{\text{abs}}^\top r \right)^\top z$$

where $A_{\text{abs}} \in \{0,1\}^{n \times m}$ is the unsigned incidence matrix ($A_{\text{abs}}[v, i] = 1$ iff node $v$ is an endpoint of edge $i$), so $\frac{1}{2} A_{\text{abs}} z$ counts how many times each node is visited. The factor $\frac{1}{2}$ accounts for the double-counting of each edge visit to its two endpoints.

**Constraints:**

$$z \geq \mathbf{C}_{\text{oriented}}(w^+ - w^-) \tag{abs-lower-1}$$

$$z \geq -\mathbf{C}_{\text{oriented}}(w^+ - w^-) \tag{abs-lower-2}$$

$$z \leq \mathbf{C}_{\text{oriented}}(w^+ - w^-) + M(1 - b) \tag{abs-upper-1}$$

$$z \leq -\mathbf{C}_{\text{oriented}}(w^+ - w^-) + M \cdot b \tag{abs-upper-2}$$

$$\boldsymbol{\tau}^\top z \leq T_{\max} \tag{time budget}$$

$$\frac{1}{2} A_{\text{abs}}\, z \geq V_{\min} \tag{visit counts}$$

$$b \in \{0,1\}^m,\quad w^+, w^-, z \in \mathbb{Z}_{\geq 0}^{k,k,m} \tag{integrality}$$

Constraints (abs-lower-1/2) and (abs-upper-1/2) jointly linearize $z = |\mathbf{C}_{\text{oriented}} w|$ via Big-M. The constant $M = 20000$ is chosen to be larger than any feasible traversal count. This MILP has $2k + 2m$ integer variables and $4m + 1 + n$ constraints.

**Pseudocode:**

```
Algorithm 1: SolveMILP(V_min, C_oriented, tau, r, T_max, A_abs, M)
─────────────────────────────────────────────────────────────────────────────
Input:  V_min ∈ Z^n, C_oriented ∈ Z^{m×k}, tau ∈ R^m, r ∈ R^n, T_max
Output: {J, T, w_signed, z_opt, flow_B, v_opt} or NULL

1:  Declare x = [w+; w-; z; b] ∈ Z^{2k+m+m}
2:  Build constraint matrix from (abs-lower-1/2), (abs-upper-1/2),
    (time budget), (visit counts)
3:  Solve: max (1/2)(A_abs^T r)^T z
           s.t. A_ub x ≤ b_ub,  x integer,  0 ≤ b ≤ 1
    (HiGHS/SCIP backend via scipy.milp)
4:  if not feasible: return NULL
5:  w_signed ← round(w+) − round(w−)
6:  z_opt    ← round(z)
7:  flow_B   ← C_oriented · w_signed
8:  v_opt    ← (1/2) A_abs · z_opt
9:  J        ← r^T v_opt
10: T        ← tau^T z_opt
11: return {J, T, w_signed, z_opt, flow_B, v_opt}
─────────────────────────────────────────────────────────────────────────────
```

### III-C.2. Two-Cycle Verification via Urgency DFS

A raw MILP solution $f_B$ guarantees that the robot visits each node at least $V_{\min}[v]$ times per cycle, but does **not** guarantee that the visits are temporally distributed to satisfy latency. A visit schedule that clusters all $V_{\min}[v]$ visits at the beginning of the cycle and then leaves the node unvisited for most of it would satisfy the integer constraint while violating the latency bound.

To verify genuine latency feasibility, we run a **backtracking Eulerian circuit search** (urgency DFS) on the directed multigraph $DG_f$ induced by $f_B$, where each $f_B[i] > 0$ contributes $f_B[i]$ directed copies of edge $i$ in the canonical direction, and $f_B[i] < 0$ contributes $|f_B[i]|$ copies in reverse.

The urgency DFS maintains cumulative time $t$, per-node last-visit time $t_v$, and a distance lower bound precomputed by Dijkstra on the condensed edge-weighted graph. At each step, among candidate next-hop edges, the DFS orders them by urgency:

$$\text{urgency}(v) = -\frac{t + \tau_{(u,v)} - t_v^{\text{last}}}{L_v}$$

(the more overdue a node is relative to its latency budget, the higher its urgency). A candidate branch is pruned immediately if any node $v$ would exceed its latency even on the fastest possible path to $v$:

$$t - t_v^{\text{last}} + d_{\min}(u, v) > L_v \implies \text{prune}$$

Once an Eulerian circuit $\mathcal{W}$ is found, it is subjected to **two-cycle verification**: the robot simulates running the circuit twice back-to-back and records all visit timestamps. For each node $v$ and each consecutive pair of visits in the double-run, the gap must satisfy:

$$\max_{i} (t_v^{(i+1)} - t_v^{(i)}) \leq L_v$$

This catches **wrap-around gaps** at the seam between two consecutive cycles, which a single-cycle check would miss.

**Pseudocode:**

```
Algorithm 2: VerifyLatency(f_B, edge_tau_dict, latency_req, nodes)
─────────────────────────────────────────────────────────────────────────────
Input:  flow f_B, travel times, latency requirements
Output: (euler_path W, violated_nodes) or NULL

1:  Build directed multigraph DG_f from f_B
2:  Check: all active nodes degree-balanced and weakly connected
    if not: return NULL
3:  Precompute dist_lb[u][v] = shortest-path distance u→v in DG_f
4:  for source in possible_sources (try up to n sources):
5:      W ← UrgencyDFS(DG_f, source, dist_lb, latency_req)
6:      if W = NULL: continue
7:      // Two-cycle verification
8:      simulate W + W (double run), record visit_times[v]
9:      T_tour ← total time of W
10:     violated ← []
11:     for each node v:
12:         gaps ← consecutive differences in visit_times[v]
13:         if max(gaps) > latency_req[v]:
14:             violated.append((v, max(gaps), latency_req[v]))
15:     if violated is empty:
16:         return (W, [])       // latency-feasible
17:     keep best W by min worst_gap
18: return (best_W, best_violated)
─────────────────────────────────────────────────────────────────────────────
```

### III-C.3. BFS+Frontier Search over the V_MIN Lattice

A single MILP call with the initial $V_{\min}$ may not yield a latency-feasible solution after DFS verification. The fundamental reason is that $V_{\min}[v] = \lceil T_{\max} / L_v \rceil$ is a loose lower bound: having a node visited $V_{\min}[v]$ times does not preclude those visits from clustering together.

We therefore treat $V_{\min} \in \mathbb{Z}^n_{\geq 0}$ as a **lattice point** and search the lattice using a priority queue (BFS + Frontier). The priority key is a lexicographic pair $(\text{cost},\ -J)$: we first minimize the total increment above the initial $V_{\min}$ (to preserve solution quality), then maximize reward $J$ among states with equal increment.

At each iteration, if the DFS confirms latency feasibility, the current $f_B$ is accepted. Otherwise, the **boundary nodes** — those nodes $v$ for which $v_{\text{opt}}[v] = V_{\min}[v]$ (visit count is tight at the MILP lower bound) — are identified and their $V_{\min}$ entries are incremented by $+1$, generating child states to push onto the frontier.

**Pseudocode:**

```
Algorithm 3: BFSFrontierSearch(G, C_oriented, tau, r, T_max, L_req)
─────────────────────────────────────────────────────────────────────────────
Output: latency-feasible flow f_B and Eulerian circuit W_B

1:  V_min_0 ← ⌈T_max / L_req⌉  element-wise
2:  frontier ← empty min-heap (key: (cost, -J))
3:  visited   ← empty set
4:
5:  Push(frontier, V_min_0, cost=0)
6:
7:  while frontier not empty:
8:      (cost, -J, uid, milp_res) ← Pop(frontier)
9:      print: "Iter, cost, J, T, edges, frontier size"
10:
11:     result ← VerifyLatency(milp_res.flow_B, ...)
12:     if result ≠ NULL and result.violated is empty:
13:         f_B ← milp_res.flow_B
14:         W_B ← result.euler_path
15:         return (f_B, W_B)            // SUCCESS
16:
17:     // Expand: find boundary nodes
18:     boundary ← {v : v_opt[v] == V_min[v]}
19:     for v in boundary:
20:         V_min_new ← V_min.copy(),  V_min_new[v] += 1
21:         if V_min_new not in visited:
22:             milp_res_new ← SolveMILP(V_min_new, ...)
23:             if milp_res_new ≠ NULL:
24:                 Push(frontier, V_min_new, cost+1)
25:             visited.add(V_min_new)
26:
27: raise RuntimeError("No latency-feasible flow found")
─────────────────────────────────────────────────────────────────────────────
```

The priority ordering $(\text{cost}, -J)$ ensures that the search is both **complete** (every lattice point is eventually visited if needed) and **reward-greedy** (among states with equal total increment, higher-reward solutions are explored first). In practice, the algorithm terminates within 1–5 MILP calls for graphs with $n \leq 15$.

---

## III-D. Cycle-Space Correction: The Algebraic Foundation

Once $f_B$ and $\mathcal{W}_B$ are established, the system enters the online monitoring phase. When edges $e_1^*, e_2^*$ (or a single edge $e^*$) become blocked, the robot requires a corrected flow $f_{\text{corr}}$ satisfying $f_{\text{corr}}[e_i^*] = 0$ while preserving all structural properties of $f_B$.

**Proposition 1 (Kirchhoff Guarantee).** For any $\beta \in \mathbb{Z}^k$, the correction vector $C_m := \mathbf{C}_{\text{oriented}} \beta$ satisfies $B_{\text{oriented}} \cdot C_m = \mathbf{0}$.

*Proof.* Each column of $\mathbf{C}_{\text{oriented}}$ is a fundamental cycle, hence in the null space of $B_{\text{oriented}}$. The product $B_{\text{oriented}} \mathbf{C}_{\text{oriented}} = 0$ column-wise, so $B_{\text{oriented}} C_m = B_{\text{oriented}} \mathbf{C}_{\text{oriented}} \beta = 0$ for all $\beta$. $\square$

Consequently, $f_{\text{corr}} = f_B + C_m$ also satisfies degree balance, since $B_{\text{oriented}} f_{\text{corr}} = B_{\text{oriented}} f_B + B_{\text{oriented}} C_m = 0 + 0 = 0$. **Degree balance is guaranteed for free by the algebraic structure** — no additional constraints are needed for it. The blocking requirement translates to a single linear equality in $\beta$:

$$(\mathbf{C}_{\text{oriented}} \beta)[j^*] = d^* := -f_B[j^*]$$

for each blocked edge index $j^*$. The challenge is to choose $\beta$ so that the **induced subgraph** $G[\text{supp}(f_{\text{corr}})]$ remains weakly connected.

---

## III-E. Single-Edge Blocking: Connectivity-Constrained MIQP

### III-E.1. Cycle-Basis Overlap Graph

To reason about connectivity of the corrected flow's induced subgraph without solving a full graph-connectivity integer program, we introduce a compact auxiliary structure.

**Definition 1 (Unsigned Cycle-Edge Incidence).** Let $\mathbf{C} \in \{0,1\}^{m \times k}$ be the unsigned version of $\mathbf{C}_{\text{oriented}}$: $\mathbf{C}[i,j] = 1$ iff edge $i$ belongs to basis cycle $j$ (regardless of orientation).

**Definition 2 (Cycle-Basis Overlap Graph).** The overlap graph $\mathcal{H} = (\{1,\ldots,k\},\, \mathcal{F})$ has a node for each basis cycle and an undirected edge $(j_1, j_2) \in \mathcal{F}$ whenever cycles $j_1$ and $j_2$ share at least one original graph edge:

$$(j_1, j_2) \in \mathcal{F} \iff \exists\, i \in \{1,\ldots,m\} : \mathbf{C}[i, j_1] = 1 \text{ and } \mathbf{C}[i, j_2] = 1$$

Two basis cycles are adjacent in $\mathcal{H}$ if and only if their respective edge-sets overlap in $G$. If a subset $\mathcal{Z} \subseteq \{1,\ldots,k\}$ of basis cycles is activated (i.e., $\beta_j \neq 0$ for $j \in \mathcal{Z}$), then the **induced subgraph** $G[\bigcup_{j \in \mathcal{Z}} \text{supp}(c_j)]$ is weakly connected in $G$ if and only if the induced subgraph $\mathcal{H}[\mathcal{Z}]$ is connected.

**Definition 3 (Anchor Set).** For a blocked edge $e^*$ with index $j^*$, the anchor set $A = \{j : \mathbf{C}[j^*, j] = 1\}$ is the set of basis cycles whose support contains $e^*$. Any correction with $C_m[j^*] = d^* \neq 0$ must activate at least one basis in $A$.

### III-E.2. Single-Commodity Flow Encoding of Induced-Subgraph Connectivity

We want to enforce: the induced subgraph $\mathcal{H}[\mathcal{Z}^*]$ of active bases is connected and contains at least one anchor in $A$. We encode this as a **single-commodity flow problem** over $\mathcal{H}$: a virtual supersource injects flow into the anchor nodes $A$, and every active node must absorb at least one unit of flow. Since flow can only travel along arcs whose both endpoints are active, every active node must be reachable from $A$ through the active sub-network of $\mathcal{H}$.

Introduce binary activity indicators $z_j \in \{0,1\}$ (active iff $z_j = 1$), virtual source flows $\phi_i \geq 0$ for each anchor $A[i]$, and arc flows $\psi_{j_1 \to j_2} \geq 0$ for each directed arc of $\mathcal{H}$:

$$\min_{\beta, z, \phi, \psi} \quad \|\mathbf{C}_{\text{oriented}} \beta\|_2^2 + \delta \sum_j z_j$$

subject to:

$$\mathbf{C}_{\text{oriented}}[j^*, :] \cdot \beta = d^* \tag{Blocking}$$

$$-M_0 z_j \leq \beta_j \leq M_0 z_j \quad \forall j \tag{Big-M linking}$$

$$\phi_i \leq (k-1)\, z_{A[i]} \quad \forall i \in A \tag{Source capacity}$$

$$\psi_{j_1 \to j_2} \leq (k-1)\, z_{j_1},\quad \psi_{j_1 \to j_2} \leq (k-1)\, z_{j_2} \quad \forall (j_1,j_2) \in \mathcal{F} \tag{Arc capacity}$$

$$\sum_{j_1 \to j} \psi_{j_1 \to j} + \mathbf{1}[j \in A]\, \phi_{\text{idx}(j)} - \sum_{j \to j_2} \psi_{j \to j_2} \geq z_j \quad \forall j \tag{Absorption}$$

The absorption constraint at node $j$ forces every active basis to receive net in-flow $\geq 1$ from the virtual supersource, which — combined with the arc capacity constraints restricting flow to active-active arcs — guarantees that $j$ is reachable from $A$ through a connected path of active bases in $\mathcal{H}$.

**Proposition 2 (Induced-Subgraph Connectivity Guarantee).** Any feasible solution $(\beta^*, z^*)$ to the above MIQP satisfies: the induced subgraph $\mathcal{H}[\mathcal{Z}^*]$, where $\mathcal{Z}^* = \{j : z_j^* = 1\}$, is connected in $\mathcal{H}$ and contains at least one anchor in $A$. Consequently, the induced subgraph $G\!\left[\bigcup_{j \in \mathcal{Z}^*} \text{supp}(c_j)\right]$ is weakly connected in $G$, and since $\text{supp}(f_{\text{corr}}) \subseteq \text{supp}(f_B) \cup \bigcup_{j \in \mathcal{Z}^*} \text{supp}(c_j)$, the induced subgraph $G[\text{supp}(f_{\text{corr}})]$ is weakly connected.

*Proof.* By the absorption constraints, every $j \in \mathcal{Z}^*$ receives at least one unit of net in-flow. By the arc capacity constraints, this flow can only traverse arcs where both endpoints are in $\mathcal{Z}^*$. The only flow source is the virtual supersource connected to the anchor nodes $A \cap \mathcal{Z}^*$. Therefore, every node in $\mathcal{Z}^*$ is reachable from $A \cap \mathcal{Z}^*$ through a path in $\mathcal{H}[\mathcal{Z}^*]$, which means $\mathcal{H}[\mathcal{Z}^*]$ is connected. The graph-connectivity claim follows from the definition of the overlap graph. $\square$

**Pseudocode:**

```
Algorithm 4: SingleEdge-ConnectivityMIQP(f_B, e*, G, C_oriented, C_matrix)
─────────────────────────────────────────────────────────────────────────────
Input:  f_B ∈ Z^m,  blocked edge e* with index j*
Output: f_corr ∈ Z^m  or  FAILURE

1:  d* ← −f_B[j*]
2:  A ← {j : C_matrix[j*, j] = 1}           // anchor set
3:  Build H: for j1,j2 in [1..k]: if col j1,j2 share an edge → add arc
4:  dir_arcs ← directed version of H (both directions per undirected edge)
5:  Declare β ∈ Z^k, z ∈ {0,1}^k, φ ∈ R^|A|_≥0, ψ ∈ R^|dir_arcs|_≥0
6:  Add constraints:
        C_oriented[j*, :] · β = d*
        −M0·z_j ≤ β_j ≤ M0·z_j              ∀j
        φ_i ≤ (k−1)·z_{A[i]}                ∀i∈A
        ψ_{j1→j2} ≤ (k−1)·min(z_{j1}, z_{j2})  ∀arcs
        Σ_in ψ + φ(if j∈A) − Σ_out ψ ≥ z_j  ∀j   // absorption
7:  Solve: min ‖C_oriented · β‖₂² + δ · Σ z_j   (SCIP)
8:  if infeasible: return FAILURE
9:  β* ← round(β.value)
10: C_m ← C_oriented · β*
11: f_corr ← f_B + C_m
12: return f_corr           // weakly connected by Prop. 2
─────────────────────────────────────────────────────────────────────────────
```

---

## III-F. Simultaneous Two-Edge Blocking: Two-Stage MIQP

When two non-adjacent active edges $e_1^* = (u_1, v_1)$ and $e_2^* = (u_2, v_2)$ (with $\{u_1, v_1\} \cap \{u_2, v_2\} = \emptyset$) are blocked simultaneously, we extend the correction to two blocking equalities:

$$(\mathbf{C}_{\text{oriented}} \beta)[j_1^*] = d_1,\quad (\mathbf{C}_{\text{oriented}} \beta)[j_2^*] = d_2$$

However, the two simultaneous constraints substantially narrow the feasible set of $\beta$, making the overlap-graph connectivity encoding both more complex and more likely to be infeasible. We instead adopt a **two-stage MIQP** that prioritizes preserving the tour duration before minimizing the correction norm, without explicit connectivity encoding. When the MIQP correction produces a disconnected induced subgraph, the Detour fallback (Section III-G) is invoked.

### III-F.1. Motivation: Tour-Time Deviation as a Quality Proxy

Define the original tour time $T_{\text{orig}} = \boldsymbol{\tau}^\top |f_B|$ and the corrected tour time $T_{\text{new}} = \boldsymbol{\tau}^\top |f_{\text{corr}}|$. A correction with small $|\Delta T| = |T_{\text{new}} - T_{\text{orig}}|$ is preferable: it implies the robot is replacing the two blocked traversals with routes of comparable cost, preserving the visit-frequency balance established by the original MILP. We therefore use $|\Delta T|$ as a first-stage criterion, deferring $\|C_m\|_2^2$ minimization to a second stage.

### III-F.2. Big-M Absolute Value Linearization

To handle $|f_{\text{corr}}[i]|$ in an integer program, introduce $z_{\text{abs}} \in \mathbb{Z}^m$ and $b \in \{0,1\}^m$:

$$z_{\text{abs}}[i] \geq f_{\text{corr}}[i], \quad z_{\text{abs}}[i] \geq -f_{\text{corr}}[i]$$
$$z_{\text{abs}}[i] \leq f_{\text{corr}}[i] + M(1-b[i]), \quad z_{\text{abs}}[i] \leq -f_{\text{corr}}[i] + M \cdot b[i]$$

with $f_{\text{corr}} = f_B + \mathbf{C}_{\text{oriented}} \beta$. Then $T_{\text{new}} = \boldsymbol{\tau}^\top z_{\text{abs}}$ and $\Delta T = T_{\text{new}} - T_{\text{orig}}$.

### III-F.3. Stage 1 — Minimize Absolute Time Deviation

$$\eta^* = \min_{\beta \in \mathbb{Z}^k,\, z_{\text{abs}} \in \mathbb{Z}^m,\, b \in \{0,1\}^m,\, \eta \in \mathbb{R}} \quad \eta$$

subject to:

$$(\mathbf{C}_{\text{oriented}} \beta)[j_1^*] = d_1,\quad (\mathbf{C}_{\text{oriented}} \beta)[j_2^*] = d_2 \tag{Blocking}$$

$$\text{Big-M absolute value constraints for } z_{\text{abs}} \equiv |f_B + \mathbf{C}_{\text{oriented}} \beta| \tag{Abs-val}$$

$$\eta \geq \boldsymbol{\tau}^\top z_{\text{abs}} - T_{\text{orig}}, \quad \eta \geq T_{\text{orig}} - \boldsymbol{\tau}^\top z_{\text{abs}} \tag{Dev bound}$$

The value $\eta^*$ is the minimum achievable absolute tour-time deviation given the two hard blocking constraints.

### III-F.4. Stage 2 — Minimize L2 Correction Norm Within Time Budget

$$\min_{\beta, z_{\text{abs}}, b, \eta} \quad \|\mathbf{C}_{\text{oriented}} \beta\|_2^2 \quad \text{subject to all Stage-1 constraints, plus:} \quad \eta \leq \eta^* + \varepsilon$$

with tolerance $\varepsilon = 0.2$ s. The two-stage structure ensures the correction is both temporally conservative and algebraically compact.

**Pseudocode:**

```
Algorithm 5: TwoEdge-TwoStageMIQP(f_B, e1*, e2*, C_oriented, tau, ε=0.2)
─────────────────────────────────────────────────────────────────────────────
Input:  f_B ∈ Z^m, blocked indices j1*, j2*, demands d1, d2
Output: f_corr ∈ Z^m  or  FAILURE

1:  T_orig ← τ^T |f_B|
2:  Declare β ∈ Z^k, z_abs ∈ Z^m, b ∈ {0,1}^m, η ∈ R
3:  Add blocking: C_oriented[j1*,:] · β = d1,  C_oriented[j2*,:] · β = d2
4:  Add Big-M abs-val constraints for z_abs = |f_B + C_oriented · β|
5:  Add: η ≥ τ^T z_abs − T_orig,   η ≥ T_orig − τ^T z_abs
6:
7:  // Stage 1
8:  η* ← Solve min η    (SCIP)
9:  if Stage-1 infeasible: return FAILURE
10:
11: // Stage 2
12: Add: η ≤ η* + ε
13: Solve min ‖C_oriented · β‖₂²    (SCIP, same variable declarations)
14: if Stage-2 infeasible: return FAILURE
15:
16: β* ← round(β.value)
17: f_corr ← f_B + C_oriented · β*
18: return f_corr
─────────────────────────────────────────────────────────────────────────────
```

---

## III-G. Connectivity Failure and Topological Detour Fallback

### III-G.1. Why Algebraic Correction Can Disconnect the Induced Subgraph

The two-stage MIQP operates in the algebraic cycle space and has no explicit awareness of global graph topology beyond the two linear constraints. We formalize the connectivity failure mode:

**Proposition 3 (Induced-Subgraph Disconnection Under L2 Correction).** There exist graphs $G$, baseline flows $f_B$, and blocked edge pairs $(e_1^*, e_2^*)$ such that the two-stage L2-minimizing correction $\beta^*$ yields a $f_{\text{corr}} = f_B + \mathbf{C}_{\text{oriented}} \beta^*$ whose induced subgraph $G[\text{supp}(f_{\text{corr}})]$ is not weakly connected.

*Proof by construction.* Consider a graph where $f_B$'s support contains a bridge-like bottleneck edge $e_b$ connecting a left subgraph $G_L$ and a right subgraph $G_R$. Blocking $e_1^* \in G_L$ and $e_2^* \in G_R$ independently requires closing a local loop in each subgraph. The L2-minimizing $\beta^*$ activates basis cycles localized entirely within $G_L$ and $G_R$ respectively, producing a correction that closes each blocked edge's traversal locally without invoking any basis cycle spanning $e_b$. If $f_B[e_b]$ subsequently reduces to zero under this correction (or the correction zeros out edges adjacent to $e_b$), the induced subgraph $G[\text{supp}(f_{\text{corr}})]$ splits into $G_L$ and $G_R$ components. $\square$

After either MIQP algorithm, we apply:

```
IsEulerian(f):
    DG ← directed multigraph from f
    balanced ← all non-isolated nodes have in_deg == out_deg
    weakly_conn ← DG restricted to active nodes is weakly connected
    return balanced AND weakly_conn
```

If `IsEulerian(f_corr)` fails, control passes to the Detour fallback.

### III-G.2. Detour Re-routing via Topological Shortest Path

The Detour approach replaces each blocked edge's missing traversal with a topological bypass in the residual graph $G' = G \setminus \{e_1^*, e_2^*\}$, building a correction that is by construction a closed directed cycle.

**Definition 4 (Detour Correction).** Let $e^* = (u, v)$ with $f_B[e^*] = c \neq 0$. Define $s = u$ if $c > 0$ (else $s = v$) and $t = v$ if $c > 0$ (else $t = u$). Let $P^* = (s = p_0, p_1, \ldots, p_\ell = t)$ be the $\tau$-weighted shortest simple path from $s$ to $t$ in $G'$. The Detour correction vector $C_m^{\text{det}} \in \mathbb{Z}^m$ is:

$$C_m^{\text{det}}[j^*] = -c, \qquad C_m^{\text{det}}[e(p_i, p_{i+1})] \mathrel{+}= |c| \cdot \text{dir}(p_i \to p_{i+1}) \quad \forall i$$

where $\text{dir}(a \to b) = +1$ if $a < b$ in the node ordering (canonical direction) and $-1$ otherwise, and $e(a,b)$ is the edge index of $(a, b)$.

**Proposition 4 (Induced-Subgraph Connectivity of Detour).** Let $f_{\text{corr}} = f_B + C_m^{\text{det}}$. The following hold:

1. $B_{\text{oriented}} \cdot C_m^{\text{det}} = \mathbf{0}$ (degree balance preserved, by Prop. 1).
2. The induced subgraph $G[\text{supp}(f_{\text{corr}})]$ satisfies:
$$G[\text{supp}(f_{\text{corr}})] = G\!\left[(\text{supp}(f_B) \setminus \{e^*\}) \cup P^*\right]$$
In particular, if $G' = G \setminus \{e^*\}$ is connected, then the induced subgraph $G[\text{supp}(f_{\text{corr}})]$ is weakly connected.

*Proof of (2).* $C_m^{\text{det}}[e^*] = -c$ zeros out $e^*$ from $f_{\text{corr}}$. For all $e \notin P^* \cup \{e^*\}$, $C_m^{\text{det}}[e] = 0$, so $f_{\text{corr}}[e] = f_B[e]$. For $e \in P^*$, $C_m^{\text{det}}[e] = \pm|c|$, which can only increase $|f_{\text{corr}}[e]|$ (adding traversals along $P^*$). Therefore $\text{supp}(f_{\text{corr}}) = (\text{supp}(f_B) \setminus \{e^*\}) \cup P^*$, giving the stated induced subgraph equality.

Connectivity: the induced subgraph $G[\text{supp}(f_B)]$ is weakly connected (it was verified Eulerian). Removing $e^*$ may disconnect it into components $\mathcal{G}_s \ni s$ and $\mathcal{G}_t \ni t$. The path $P^*$ reconnects $s$ to $t$ via $G'$, so $G[(\text{supp}(f_B) \setminus \{e^*\}) \cup P^*]$ is weakly connected. $\square$

For simultaneous two-edge blocking, the corrections are applied sequentially on the same $G' = G \setminus \{e_1^*, e_2^*\}$. At each step, the candidate detour paths are tested for combined Eulerian validity:

**Pseudocode:**

```
Algorithm 6: DetourFallback(f_B, blocked_list, G, tau, K=10)
─────────────────────────────────────────────────────────────────────────────
Input:  f_B ∈ Z^m, blocked_list = [(j*, c, s, t), ...]  (≤ 2 entries)
Output: f_corr ∈ Z^m  or  FAILURE

1:  G' ← G with all blocked edges removed, edge weights ← tau
2:  total_Cm ← 0^m
3:
4:  for (j*, c, s, t) in blocked_list:
5:      if not Connected(G', s, t): continue    // bridge; skip
6:      candidates ← YenKSP(G', s, t, weight=tau, K=K)
7:      best_Cm_i ← candidates[0]  // fallback: shortest path
8:      best_euler ← False
9:
10:     for path P in candidates:
11:         Cm_i ← ZeroVector(m)
12:         Cm_i[j*] ← −c
13:         for each step (a → b) in P:
14:             Cm_i[edge_idx(a,b)] += |c| · (1 if a<b else −1)
15:         test_flow ← f_B + total_Cm + Cm_i
16:         if IsEulerian(test_flow) and not best_euler:
17:             best_Cm_i ← Cm_i
18:             best_euler ← True
19:             break
20:
21:     total_Cm += best_Cm_i
22:
23: f_corr ← f_B + total_Cm
24: if IsEulerian(f_corr): return f_corr
25: else: return FAILURE
─────────────────────────────────────────────────────────────────────────────
```

---

## III-H. Complete Online Recovery Pipeline

We integrate all components into the full online recovery procedure. The pipeline tries the algebraic MIQP correction first (faster, potentially lower-norm correction); upon Eulerian failure, it cascades to the topological Detour (provably connectivity-preserving).

```
Algorithm 7: EdgeBlockingRecovery(f_B, W_B, blocked_edges, G, C_oriented, tau)
─────────────────────────────────────────────────────────────────────────────
Input:  f_B ∈ Z^m,  W_B (original Eulerian circuit)
        blocked_edges: 1 or 2 edges with indices and flow values
Output: (f_corr, W_corr, method)

─── Case A: single edge e* blocked ─────────────────────────────────────────
1:  f_corr ← SingleEdge-ConnectivityMIQP(f_B, e*, G, C_oriented, C_matrix)
2:  if f_corr ≠ FAILURE and IsEulerian(f_corr):
3:      W_corr ← GuidedFleurys(f_corr, W_B)
4:      return (f_corr, W_corr, "MIQP+Connectivity")
5:  f_corr ← DetourFallback(f_B, [e*], G, tau)
6:  if f_corr ≠ FAILURE:
7:      W_corr ← GuidedFleurys(f_corr, W_B)
8:      return (f_corr, W_corr, "Detour")
9:  return (f_B, W_B, "NO_SOLUTION")        // e* is a bridge in G

─── Case B: two non-adjacent edges e1*, e2* blocked ─────────────────────────
1:  f_corr ← TwoEdge-TwoStageMIQP(f_B, e1*, e2*, C_oriented, tau)
2:  if f_corr ≠ FAILURE and IsEulerian(f_corr):
3:      W_corr ← GuidedFleurys(f_corr, W_B)
4:      return (f_corr, W_corr, "TwoStage-MIQP")
5:  f_corr ← DetourFallback(f_B, [e1*, e2*], G, tau)
6:  if f_corr ≠ FAILURE:
7:      W_corr ← GuidedFleurys(f_corr, W_B)
8:      return (f_corr, W_corr, "Detour")
9:  return (f_B, W_B, "NO_SOLUTION")
─────────────────────────────────────────────────────────────────────────────
```

---

## III-I. Guided Fleury's Algorithm for Circuit Execution

Given a corrected flow $f_{\text{corr}}$ that has passed the Eulerian check, the final step produces a concrete traversal sequence $\mathcal{W}_{\text{corr}}$ for execution.

### III-I.1. Corrected Multigraph Construction

Define the corrected directed multigraph:

$$DG_{\text{corr}} = \bigcup_{i:\, f_{\text{corr}}[i] > 0} \{u_i \to v_i\}^{f_{\text{corr}}[i]} \;\cup\; \bigcup_{i:\, f_{\text{corr}}[i] < 0} \{v_i \to u_i\}^{|f_{\text{corr}}[i]|}$$

where $\{a \to b\}^k$ denotes $k$ directed copies of the arc $a \to b$.

**Lemma 1.** Since $f_{\text{corr}}$ is Eulerian, $DG_{\text{corr}}$ is a directed Eulerian multigraph. Any Eulerian circuit on $DG_{\text{corr}}$ traverses each arc type $(u_i \to v_i)$ exactly $|f_{\text{corr}}[i]|$ times, giving a signed traversal count equal to $f_{\text{corr}}$.

*Proof.* $DG_{\text{corr}}$ has exactly $|f_{\text{corr}}[i]|$ copies of each directed arc. An Eulerian circuit on a multigraph uses each arc (including all copies) exactly once. The signed count follows from the direction assignment. $\square$

### III-I.2. The Guided Fleury's Algorithm

Standard Fleury's algorithm selects any non-bridge edge at each step. We augment it with a **sequence-priority** oracle derived from $\mathcal{W}_B$: at each node $v$, the algorithm preferentially selects the outgoing edge whose destination matches the next step of $\mathcal{W}_B$ departing from $v$. A bookmark index $\texttt{seq\_idx}$ tracks the current position in $\mathcal{W}_B$, advancing whenever guidance is successfully followed.

The non-bridge check uses the standard Fleury's criterion for directed multigraphs: edge $(u \to v)$ is safe to take if, after its removal, (a) node $v$ still has at least one outgoing arc, and (b) the remaining graph restricted to active nodes is weakly connected.

**Pseudocode:**

```
Algorithm 8: GuidedFleurys(f_corr, W_B, edge_list, edge_to_idx)
─────────────────────────────────────────────────────────────────────────────
Input:  f_corr ∈ Z^m, original circuit W_B
Output: W_corr (Eulerian circuit on DG_corr)

1:  DG_corr ← Build multigraph from f_corr (with directed copies)
2:  DG_curr ← copy(DG_corr)
3:  curr ← W_B[0].source,   seq_idx ← 0,   W_corr ← []
4:
5:  while DG_curr has edges:
6:      out_edges ← OutArcs(DG_curr, curr)
7:
8:      // ── Fleury non-bridge filtering ──────────────────────────────────
9:      if |out_edges| = 1:
10:         valid ← out_edges
11:     else:
12:         valid ← []
13:         for arc (curr → v, key) in out_edges:
14:             DG_curr.remove(curr → v, key)
15:             is_last ← (|DG_curr.edges| = 0)
16:             safe ← is_last
17:                     OR (OutDeg(DG_curr, v) > 0
18:                         AND WeaklyConnected(DG_curr, active_nodes_only))
19:             if safe: valid.append((curr → v, key))
20:             DG_curr.restore(curr → v, key)
21:         if valid empty: valid ← out_edges    // safety fallback
22:
23:     // ── Original-sequence guidance ───────────────────────────────────
24:     chosen ← None
25:     for i from seq_idx to |W_B|−1:
26:         if W_B[i].source ≠ curr: continue
27:         pref_dst ← W_B[i].target
28:         for e in valid:
29:             if e.target = pref_dst:
30:                 chosen ← e,   seq_idx ← i+1
31:                 break
32:         if chosen ≠ None: break
33:
34:     // ── Fallback to first valid arc ──────────────────────────────────
35:     if chosen = None:
36:         chosen ← valid[0]
37:
38:     DG_curr.remove(chosen)
39:     W_corr.append((curr, chosen.target))
40:     curr ← chosen.target
41:
42: return W_corr
─────────────────────────────────────────────────────────────────────────────
```

**Theorem 1 (Correctness of Algorithm 8).** If $f_{\text{corr}}$ passes IsEulerian, Algorithm 8 terminates and returns $\mathcal{W}_{\text{corr}}$ such that:

1. $\mathcal{W}_{\text{corr}}$ is a continuous directed closed walk that returns to its start node.
2. For each original graph edge $e_i$, $\mathcal{W}_{\text{corr}}$ traverses $e_i$ exactly $|f_{\text{corr}}[i]|$ times in the direction $\text{sign}(f_{\text{corr}}[i])$.
3. Neither blocked edge $e_1^*, e_2^*$ appears in $\mathcal{W}_{\text{corr}}$.

*Proof.*
**Claim 1.** Fleury's algorithm on a directed Eulerian multigraph always selects a non-bridge arc except when forced (single out-arc). Since $DG_{\text{corr}}$ is Eulerian and balanced, removing a non-bridge arc leaves a balanced, weakly connected digraph (possibly with one node of imbalance, which is the current node with one fewer out-arc — this is corrected upon return). The algorithm cannot get stuck: at every step, at least one outgoing arc from $\texttt{curr}$ leads to a valid continuation. The walk closes at the start node because the final arc is always forced (single out-arc returning to start).

**Claim 2.** By Lemma 1, $DG_{\text{corr}}$ contains exactly $|f_{\text{corr}}[i]|$ directed copies of each arc type. Algorithm 8 visits every arc in $DG_{\text{corr}}$ exactly once (Fleury's removes each arc after traversal). Therefore the signed traversal count equals $f_{\text{corr}}$.

**Claim 3.** Since $f_{\text{corr}}[e_1^*] = f_{\text{corr}}[e_2^*] = 0$ (by the blocking constraint), neither blocked edge has any copy in $DG_{\text{corr}}$, and thus cannot appear in $\mathcal{W}_{\text{corr}}$. $\square$

---

## III-J. Complexity Summary

| Component | Complexity | Remarks |
|-----------|-----------|---------|
| MILP (Algorithm 1) | SCIP ILP; $2k{+}2m$ vars | Sub-second for $k \leq 20$, $m \leq 30$ |
| BFS+Frontier (Algorithm 3) | $O(2^n)$ worst case | Terminates in $\leq 5$ iters in practice |
| Urgency DFS (Algorithm 2) | $O(n \cdot E_f!)$ worst case; fast with pruning | Timeout available |
| Overlap graph build | $O(k^2 m)$ | One-time offline preprocessing |
| Single-Edge MIQP (Algorithm 4) | SCIP; $k{+}k{+}|A|{+}2|\mathcal{F}|$ vars | $< 2$ s for $k \leq 20$ |
| Two-Stage MIQP (Algorithm 5) | SCIP × 2; $k{+}2m$ vars per stage | $< 5$ s per stage |
| Detour fallback (Algorithm 6) | $O(K(m + n \log n))$ per blocked edge | $K=10$; Yen's KSP |
| Eulerian check | $O(n + m)$ | Standard BFS |
| Guided Fleury's (Algorithm 8) | $O(|E_{\text{corr}}|^2)$ | Anti-bridge check per step |

---

## III-K. Theoretical Properties Summary

| Property | Guarantee | Condition |
|----------|-----------|-----------|
| Degree balance of $f_{\text{corr}}$ | Always | Proposition 1 (Kirchhoff) |
| Induced-subgraph connected, Alg. 4 | Always | Proposition 2 (overlap-graph flow) |
| Induced-subgraph connected, Alg. 6 | Always | Proposition 4; $G'$ connected |
| Alg. 5 may disconnect induced subgraph | Possible failure | Proposition 3 (handled by Detour) |
| No blocked edges in $\mathcal{W}_{\text{corr}}$ | Always | Theorem 1, Claim 3 |
| $\mathcal{W}_{\text{corr}}$ edge counts match $f_{\text{corr}}$ | Always | Theorem 1, Claim 2 |
| $\mathcal{W}_{\text{corr}}$ is a closed walk | Always | Theorem 1, Claim 1 |
| Minimum detour time increase | Alg. 6 only | Dijkstra on $G'$ with $\tau$ weights |

The key asymmetry between the algebraic (MIQP) and topological (Detour) approaches is captured by Propositions 2, 3, and 4: the single-edge connectivity MIQP (Algorithm 4) enforces induced-subgraph connectivity through the overlap-graph flow constraint; the two-edge MIQP (Algorithm 5) does not and can fail; the Detour fallback (Algorithm 6) directly maintains induced-subgraph connectivity by construction via Dijkstra's shortest path.

---

*End of Methodology Section.*
