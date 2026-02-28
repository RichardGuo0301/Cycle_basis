"""
Cycle Basis Weighted Travel Optimization
Compares Method A (|C|@|x|) vs Method B (|C@x|)
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from cycle_basis_demo import create_random_polygon_graph, get_oriented_incidence_matrix
from horton_gram_mcb import find_horton_gram_mcb

np.random.seed(42)

# === 1. Create Graph ===
G_directed, G_undirected = create_random_polygon_graph(12, 10, seed=100)
n, m = G_undirected.number_of_nodes(), G_undirected.number_of_edges()
nodes = sorted(G_undirected.nodes())
node_to_idx = {v: i for i, v in enumerate(nodes)}
print(f"Graph: n={n}, m={m}, cycle_rank={m-n+1}")

# === 2. Eulerian Analysis ===
# Check if original graph is Eulerian (even degrees at all nodes)
degrees = dict(G_undirected.degree())
odd_degree_nodes = [v for v, d in degrees.items() if d % 2 == 1]
n_odd = len(odd_degree_nodes)
print(f"Odd-degree nodes: {n_odd} → {'Eulerian' if n_odd <= 2 else 'NOT Eulerian'}")

# === 3. Compute Minimum Cycle Basis ===
# Find minimum cycle basis using Horton-Gram algorithm
edge_list = sorted([tuple(sorted(e)) for e in G_undirected.edges()])
edge_to_idx = {e: i for i, e in enumerate(edge_list)}
basis_meta, _ = find_horton_gram_mcb(G_undirected, edge_list)
k = len(basis_meta)

# Build cycle matrix C (m x k): C[i,j] = 1 if edge i is in cycle j
C_matrix = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    for i in range(m):
        if (c_data['bv'] >> i) & 1:
            C_matrix[i, j] = 1
print(f"Cycle Basis: {k} cycles")

# === 4. Oriented Cycle Matrix (Aligned with edge_list) ===
# We define C_oriented (m x k) corresponding rigidly to edge_list
# For each undirected edge e=(u,v) in edge_list (where u<v):
#   Entry is +1 if cycle traverses u->v
#   Entry is -1 if cycle traverses v->u
#   Entry is 0 if edge not in cycle

C_oriented = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    # Reconstruct cycle path
    cycle_edges = [edge_list[i] for i in range(m) if C_matrix[i, j] == 1]
    cycle_subgraph = nx.Graph(cycle_edges)
    try:
        cycle_nodes = [e[0] for e in nx.find_cycle(cycle_subgraph)]
    except:
        cycle_nodes = list(cycle_subgraph.nodes())
    
    # Map cycle direction to edge reference direction
    for i in range(len(cycle_nodes)):
        u_curr, v_curr = cycle_nodes[i], cycle_nodes[(i + 1) % len(cycle_nodes)]
        
        # Find this edge in edge_list
        # edge_list contains sorted tuples (u, v) where u < v
        u, v = sorted((u_curr, v_curr))
        if (u, v) in edge_to_idx:
            idx = edge_to_idx[(u, v)]
            
            # Determine orientation
            # Reference direction is u -> v (small to large)
            if u_curr == u and v_curr == v:
                # Cycle goes u->v, matches reference
                C_oriented[idx, j] = +1
            else:
                # Cycle goes v->u, opposite to reference
                C_oriented[idx, j] = -1

# Verify B_oriented @ C_oriented = 0
# Construct B_oriented (n x m) for the same reference directions
B_oriented = np.zeros((n, m), dtype=int)
for j, (u, v) in enumerate(edge_list):
    # Ref direction u->v.  Flow leaves u (-1), enters v (+1)
    B_oriented[node_to_idx[u], j] = -1
    B_oriented[node_to_idx[v], j] = +1

print(f"B_oriented @ C_oriented = 0: {'✓' if np.allclose(B_oriented @ C_oriented, 0) else '✗'}")


# === 5. Setup Parameters ===
# Absolute incidence matrix: A_abs[i,j] = 1 if node i is incident to edge j
A_abs = np.zeros((n, m), dtype=int)
for j, (u, v) in enumerate(edge_list):
    A_abs[node_to_idx[u], j] = A_abs[node_to_idx[v], j] = 1

np.random.seed(42)
r = np.round(np.random.uniform(1, 10, n), 1)    # Node rewards
tau = np.round(np.random.uniform(1, 5, m), 1)   # Edge travel times

# New latency-based constraint
T_max = 200
V_MIN = 4
LATENCY_MAX = T_max / V_MIN  # Maximum time gap between consecutive visits = 50.0

print(f"Latency constraint: max gap = {LATENCY_MAX:.1f} (T_max={T_max} / V_MIN={V_MIN})")

# === 6. Method A: |C|@|x| (Unsigned cycle choice) ===
# Maximize reward subject to: time ≤ T_max, V_MIN ≤ node_visits
c_obj = 0.5 * C_matrix.T @ A_abs.T @ r
A_node = A_abs @ C_matrix
A_ub = np.vstack([(tau @ C_matrix).reshape(1,-1), -A_node])
b_ub = np.hstack([[T_max], -2*V_MIN*np.ones(n)])

result = milp(c=-c_obj, constraints=LinearConstraint(A_ub, -np.inf, b_ub),
              bounds=Bounds(lb=np.ones(k), ub=np.full(k, np.inf)), integrality=np.ones(k, dtype=int))

if result.success:
    w_opt = np.round(result.x).astype(int)
    x_opt_A = C_matrix @ w_opt
    v_opt_A = 0.5 * A_abs @ x_opt_A
    J_A, T_A = r @ v_opt_A, tau @ x_opt_A
    print(f"\n[Method A] J={J_A:.0f}, T={T_A:.0f}, edges={np.sum(x_opt_A)}")
else:
    print(f"\n[Method A] Optimization Failed: {result.message}")
    J_A, T_A, x_opt_A = 0, 0, np.zeros(m)


# === 7. Method B: |C@x| (Signed cycle choice with cancellation) ===
# Use w = w+ - w- decomposition to allow negative cycle weights
# Variables: [w+ (k), w- (k), z (m), b (m)]
# z = |C_oriented @ (w+ - w-)|
# b is binary auxiliary var for Big-M

n_vars = 2*k + m + m
M = 20000  # Big-M constant

# Objective: maximize reward = 0.5 * (A_abs.T @ r) @ z
c_obj_B = np.zeros(n_vars)
c_obj_B[2*k:2*k+m] = 0.5 * A_abs.T @ r

# Constraints
# (1) z >= f  =>  -C@w+ + C@w- + z >= 0
A1 = np.hstack([-C_oriented, C_oriented, np.eye(m), np.zeros((m, m))])
# (2) z >= -f =>   C@w+ - C@w- + z >= 0
A2 = np.hstack([C_oriented, -C_oriented, np.eye(m), np.zeros((m, m))])
# (3) z <= f + M*(1-b)  =>  -C@w+ + C@w- + z + M*b <= M
A3 = np.hstack([-C_oriented, C_oriented, np.eye(m), M * np.eye(m)])
# (4) z <= -f + M*b     =>   C@w+ - C@w- + z - M*b <= 0
A4 = np.hstack([C_oriented, -C_oriented, np.eye(m), -M * np.eye(m)])

# Time constraint: tau @ z <= T_max
A_time_B = np.zeros((1, n_vars)); A_time_B[0, 2*k:2*k+m] = tau

# Node visit constraints: V_MIN <= 0.5 * A_abs @ z (no upper bound)
A_node_B = np.zeros((n, n_vars)); A_node_B[:, 2*k:2*k+m] = A_abs

# Stack all constraints
# Note: milp uses A_ub @ x <= b_ub
# So we convert >= constraints to <= by multiplying by -1
A_ub_B = np.vstack([
    -A1,            # A1 >= 0  ->  -A1 <= 0
    -A2,            # A2 >= 0  ->  -A2 <= 0
    A3,             # A3 <= M
    A4,             # A4 <= 0
    A_time_B,       # time <= T_max
    -A_node_B       # v >= V_MIN (no upper bound)
])

b_ub_B = np.hstack([
    np.zeros(m),                # -A1
    np.zeros(m),                # -A2
    M * np.ones(m),             # A3
    np.zeros(m),                # A4
    [T_max],                    # time
    -2 * V_MIN * np.ones(n)    # min visits only
])

# Bounds
lb_B = np.zeros(n_vars)
ub_B = np.full(n_vars, np.inf)
ub_B[2*k+m:] = 1  # b is binary {0, 1}

result_B = milp(c=-c_obj_B, constraints=LinearConstraint(A_ub_B, -np.inf, b_ub_B),
                bounds=Bounds(lb=lb_B, ub=ub_B), 
                integrality=np.ones(n_vars, dtype=int))

if result_B.success:
    sol_B = np.round(result_B.x).astype(int)  # Critical fix: Rounding
    w_plus, w_minus = sol_B[:k], sol_B[k:2*k]
    z_opt = sol_B[2*k:2*k+m]
    # b_opt = sol_B[2*k+m:]
    
    w_signed = w_plus - w_minus
    v_opt_B = 0.5 * A_abs @ z_opt
    J_B, T_B = r @ v_opt_B, tau @ z_opt
    
    print(f"[Method B] J={J_B:.0f}, T={T_B:.0f}, edges={np.sum(z_opt)}")
    print(f"  w_signed (first 5): {w_signed[:5]}...")
else:
    print(f"[Method B] Optimization Failed: {result_B.message}")
    J_B, T_B, z_opt = 0, 0, np.zeros(m, dtype=int)
    w_signed = np.zeros(k, dtype=int)

# Verify z = |flow|
flow_B = C_oriented @ w_signed
abs_flow = np.abs(flow_B)
slack = z_opt - abs_flow

print(f"\n[Method B Verification]")
print(f"  Total slack (unnecessary traversals) = {np.sum(slack)}")
if np.sum(slack) > 0:
    print(f"  Note: Method B traverses some edges more than required by cycle selection alone.")
    print(f"        This usually happens to satisfy Eulerian conditions implicitly or constraints.")

# Verify Eulerian Condition for Method B (DIRECTED)
# Use signed flow f = C_oriented @ w, NOT |f|
# Kirchhoff's law B @ C_oriented = 0 guarantees in_degree = out_degree

flow_B = C_oriented @ w_signed  # Signed flow (not absolute value!)

DG_B = nx.MultiDiGraph()
DG_B.add_nodes_from(nodes)
for i, (u, v) in enumerate(edge_list):
    f = flow_B[i]
    if f > 0:
        # Positive flow: u -> v
        for _ in range(f):
            DG_B.add_edge(u, v)
    elif f < 0:
        # Negative flow: v -> u (reverse direction)
        for _ in range(abs(f)):
            DG_B.add_edge(v, u)

# Check directed Eulerian conditions
in_deg = dict(DG_B.in_degree())
out_deg = dict(DG_B.out_degree())
non_iso_B = [v for v in nodes if in_deg[v] + out_deg[v] > 0]
balanced = all(in_deg[v] == out_deg[v] for v in non_iso_B)
is_weakly_conn = nx.is_weakly_connected(DG_B.subgraph(non_iso_B)) if non_iso_B else True

print(f"  Directed Eulerian check: balanced={balanced}, weakly_connected={is_weakly_conn}")
if balanced and is_weakly_conn:
    print(f"  ✓ Method B forms a valid DIRECTED Eulerian Circuit!")
    print(f"  Total directed edges: {DG_B.number_of_edges()}")
    
    # === Generate Eulerian Path (like animate_cycles.py) ===
    # Try to generate Eulerian circuit with fallback
    try:
        if nx.is_eulerian(DG_B):
            euler_path_B = list(nx.eulerian_circuit(DG_B, source=nodes[0]))
        else:
            print("  Warning: Directed graph not Eulerian. Using edge_dfs.")
            euler_path_B = list(nx.edge_dfs(DG_B, source=nodes[0]))
    except Exception as e:
        print(f"  Error generating path: {e}")
        euler_path_B = []
    
    # Build edge time lookup (handles both directions)
    edge_tau_dict = {}
    for i, (u, v) in enumerate(edge_list):
        edge_tau_dict[(u, v)] = tau[i]
        edge_tau_dict[(v, u)] = tau[i]
    
    # Track cumulative time and node visit times
    cumulative_time = 0.0
    node_visit_times = {v: [] for v in nodes}
    
    if euler_path_B:
        start_node = euler_path_B[0][0]
        node_visit_times[start_node].append(0.0)
    
    # Traverse the ORDERED sequence
    for (u, v) in euler_path_B:
        cumulative_time += edge_tau_dict[(u, v)]
        node_visit_times[v].append(cumulative_time)
    
    # Check latency constraint
    violated_nodes = []
    for node, times in node_visit_times.items():
        if len(times) >= 2:
            gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
            max_gap = max(gaps)
            if max_gap > LATENCY_MAX:
                violated_nodes.append((node, max_gap))
    
    print(f"\n  === Latency Verification ===")
    print(f"  Sequence length: {len(euler_path_B)} steps, total time: {cumulative_time:.1f}")
    if not violated_nodes:
        print(f"  ✓ All nodes satisfy latency constraint ≤ {LATENCY_MAX:.1f}")
    else:
        print(f"  ✗ Latency violated for {len(violated_nodes)} nodes:")
        for node, max_gap in sorted(violated_nodes, key=lambda x: -x[1])[:5]:
            print(f"    Node {node}: max gap = {max_gap:.1f} > {LATENCY_MAX:.1f}")
else:
    print(f"  ✗ Method B does NOT form a valid directed Eulerian circuit.")

# === 8. Comparison ===
print(f"\n{'='*50}")
print(f"Method A: J={J_A:.0f}, edges={np.sum(x_opt_A)}")
print(f"Method B: J={J_B:.0f}, edges={np.sum(z_opt)}")
print(f"Improvement: {(J_B-J_A)/J_A*100:.1f}%")
print(f"{'='*50}")

# === 9. Verify Eulerian for Optimal Solution ===
# Create multigraph with edge multiplicities from Method A
MG = nx.MultiGraph()
MG.add_nodes_from(nodes)
for i, (u, v) in enumerate(edge_list):
    for _ in range(x_opt_A[i]):
        MG.add_edge(u, v)

opt_degrees = dict(MG.degree())
opt_odd = [v for v, d in opt_degrees.items() if d % 2 == 1]
non_isolated = [v for v, d in opt_degrees.items() if d > 0]
is_connected = nx.is_connected(MG.subgraph(non_isolated)) if non_isolated else True

print(f"\nOptimal multigraph: odd_nodes={len(opt_odd)}, connected={'✓' if is_connected else '✗'}")

if len(opt_odd) == 0 and is_connected:
    print("✓ Eulerian CIRCUIT exists!")
    
    # Hierholzer's algorithm: reconstruct Eulerian circuit
    euler_edges = list(nx.eulerian_circuit(MG, source=0))
    edge_tau = {(u,v): tau[edge_to_idx[tuple(sorted((u,v)))]] for u,v in edge_list}
    edge_tau.update({(v,u): t for (u,v), t in edge_tau.items()})
    
    # Track node visit times for latency analysis
    cumulative_time = 0.0
    node_visits = {v: [0.0] if v == 0 else [] for v in nodes}
    for u, v in euler_edges:
        cumulative_time += edge_tau[(u, v)]
        node_visits[v].append(cumulative_time)
    
    print(f"  Path length: {len(euler_edges)} edges, total time: {cumulative_time:.1f}")
    
    # Compute maximum time gap between consecutive visits to any node
    max_gaps = []
    for v in nodes:
        times = sorted(node_visits[v])
        if len(times) >= 2:
            max_gaps.append(max(times[i+1] - times[i] for i in range(len(times)-1)))
    print(f"  Max latency gap: {max(max_gaps):.1f}")
elif len(opt_odd) == 2 and is_connected:
    print(f"✓ Eulerian PATH exists (start/end: {opt_odd})")
else:
    print(f"✗ NOT Eulerian (needs {len(opt_odd)//2} strokes)")

# === 10. Visualization ===
pos = nx.get_node_attributes(G_undirected, 'pos') or nx.spring_layout(G_undirected, seed=42)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
nx.draw_networkx_nodes(G_undirected, pos, node_color=plt.cm.Reds(r/r.max()), node_size=400, ax=ax)
nx.draw_networkx_labels(G_undirected, pos, {v: f"{v}\n{r[node_to_idx[v]]:.1f}" for v in nodes}, font_size=7, ax=ax)
nx.draw_networkx_edges(G_undirected, pos, width=1.5, ax=ax)
ax.set_title(f"Graph (n={n}, m={m}, k={k})", fontweight='bold'); ax.axis('off')

ax = axes[0, 1]
colors = plt.cm.tab20(np.linspace(0, 1, k))
nx.draw_networkx_nodes(G_undirected, pos, node_color='lightgray', node_size=300, ax=ax)
nx.draw_networkx_labels(G_undirected, pos, font_size=9, ax=ax)
for j in range(k):
    edges = [(edge_list[i][0], edge_list[i][1]) for i in range(m) if C_matrix[i, j]]
    nx.draw_networkx_edges(G_undirected.edge_subgraph(edges), pos, edge_color=[colors[j]], width=2, alpha=0.7, ax=ax)
ax.set_title(f"Cycle Basis ({k} cycles)", fontweight='bold'); ax.axis('off')

# Method A plot
ax = axes[1, 0]
nx.draw_networkx_nodes(G_undirected, pos, node_color=plt.cm.Blues(v_opt_A/v_opt_A.max()), node_size=400, ax=ax)
nx.draw_networkx_labels(G_undirected, pos, {v: f"{v}\n{int(v_opt_A[node_to_idx[v]])}" for v in nodes}, font_size=7, ax=ax)
# Draw unvisited edges as dashed gray
unvisited_A = [(u, v) for u, v in G_undirected.edges() if x_opt_A[edge_to_idx[tuple(sorted((u,v)))]] == 0]
visited_A = [(u, v) for u, v in G_undirected.edges() if x_opt_A[edge_to_idx[tuple(sorted((u,v)))]] > 0]
if unvisited_A:
    nx.draw_networkx_edges(G_undirected, pos, edgelist=unvisited_A, width=1, edge_color='lightgray', style='dashed', ax=ax)
if visited_A:
    widths_A = [x_opt_A[edge_to_idx[tuple(sorted(e))]]/max(x_opt_A)*5+0.5 for e in visited_A]
    nx.draw_networkx_edges(G_undirected, pos, edgelist=visited_A, width=widths_A, edge_color='blue', alpha=0.8, ax=ax)
ax.set_title(f"Method A: |C|@|x|\nJ={J_A:.0f}, T={T_A:.0f}, edges={np.sum(x_opt_A)}", fontweight='bold'); ax.axis('off')

# Method B plot
ax = axes[1, 1]
nx.draw_networkx_nodes(G_undirected, pos, node_color=plt.cm.Greens(v_opt_B/v_opt_B.max()), node_size=400, ax=ax)
nx.draw_networkx_labels(G_undirected, pos, {v: f"{v}\n{int(v_opt_B[node_to_idx[v]])}" for v in nodes}, font_size=7, ax=ax)
# Draw unvisited edges as dashed gray
unvisited_B = [(u, v) for u, v in G_undirected.edges() if z_opt[edge_to_idx[tuple(sorted((u,v)))]] == 0]
visited_B = [(u, v) for u, v in G_undirected.edges() if z_opt[edge_to_idx[tuple(sorted((u,v)))]] > 0]
if unvisited_B:
    nx.draw_networkx_edges(G_undirected, pos, edgelist=unvisited_B, width=1, edge_color='lightgray', style='dashed', ax=ax)
if visited_B:
    widths_B = [z_opt[edge_to_idx[tuple(sorted(e))]]/max(z_opt)*5+0.5 for e in visited_B]
    nx.draw_networkx_edges(G_undirected, pos, edgelist=visited_B, width=widths_B, edge_color='green', alpha=0.8, ax=ax)
conn_status = "Connected" if is_weakly_conn else f"DISCONNECTED"
ax.set_title(f"Method B: |C@x|\nJ={J_B:.0f}, T={T_B:.0f}, edges={np.sum(z_opt)}\n{conn_status}", fontweight='bold'); ax.axis('off')

plt.tight_layout()
plt.savefig("polygon_travel_optimization.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: polygon_travel_optimization.png")
plt.close()
