"""
[Comparison Method B] MIQP Cycle-Basis Repair — Reference Baseline
====================================================================
Repairs a blocked flow using the existing Cycle-Basis MIQP approach:
finds beta in cycle space such that Cm = C_oriented @ beta, minimizing
||Cm||_2^2 subject to |ΔT| ≤ 0.2s.

Prints:  MILP reward, MILP runtime, MIQP runtime,
         ⟨flow_B, Cm⟩, edge changes, node changes, ΔT
"""

import numpy as np
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds
import cvxpy as cp
import time
import osmnx as ox
from shapely.geometry import LineString
from horton_gram_mcb import find_horton_gram_mcb

if not hasattr(np, 'alltrue'):
    np.alltrue = np.all

# ─── toggle which campus to use ───────────────────────────────────────────────
USE_LEHIGH = False   # set True for Lehigh, False for ASU
# ─────────────────────────────────────────────────────────────────────────────

# ==============================
# 1) Graph builders
# ==============================
def build_asu_graph():
    point = (33.3055, -111.6815)
    G = ox.graph_from_point(point, dist=800, network_type='drive_service')
    try:
        G_undirected = ox.utils_graph.get_undirected(G)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G)

    nodes_1649 = [n for n in G_undirected.nodes() if str(n).startswith("1649")]
    G_undirected.remove_nodes_from(nodes_1649)
    n_5259, n_8310 = 5259118345, 8310002397
    if n_5259 in G_undirected:
        queue, visited, sub = [n_5259], {n_5259, n_8310}, {n_5259}
        while queue:
            curr = queue.pop(0)
            for nbr in G_undirected.neighbors(curr):
                if nbr not in visited:
                    visited.add(nbr); sub.add(nbr); queue.append(nbr)
        G_undirected.remove_nodes_from(sub)

    G_proj = ox.project_graph(G_undirected)
    G_cons = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)
    G_und  = ox.project_graph(G_cons, to_crs="EPSG:4326")
    try:
        G_undirected = ox.utils_graph.get_undirected(G_und)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G_und)

    if G_undirected.has_edge(19, 102):
        G_undirected.remove_edge(19, 102)
        if 102 in G_undirected:
            G_undirected.remove_nodes_from(list(nx.node_connected_component(G_undirected, 102)))
    if G_undirected.has_edge(28, 29):
        G_undirected.remove_edge(28, 29)
        if 29 in G_undirected:
            G_undirected.remove_nodes_from(list(nx.node_connected_component(G_undirected, 29)))

    for u, v in [(15, 13), (100, 98), (13, 31), (31, 32)]:
        if u in G_undirected and v in G_undirected:
            x1,y1 = G_undirected.nodes[u]['x'],G_undirected.nodes[u]['y']
            x2,y2 = G_undirected.nodes[v]['x'],G_undirected.nodes[v]['y']
            G_undirected.add_edge(u,v,key=0,length=ox.distance.great_circle(y1,x1,y2,x2),
                                  geometry=LineString([(x1,y1),(x2,y2)]))

    x_max = -111.670
    preserve_bbox = (33.3047, 33.3095, -111.6808, -111.6720)
    def in_box(lat,lon):
        s,n,w,e = preserve_bbox; return (s<=lat<=n) and (w<=lon<=e)
    while True:
        to_rm = [nd for nd,d in G_undirected.degree()
                 if d < 2 and not in_box(G_undirected.nodes[nd].get('y',0),
                                          G_undirected.nodes[nd].get('x',0))]
        to_rm += [nd for nd in G_undirected.nodes()
                  if nd not in to_rm
                  and G_undirected.nodes[nd].get('x',0) > x_max
                  and not in_box(G_undirected.nodes[nd].get('y',0),
                                  G_undirected.nodes[nd].get('x',0))]
        if not to_rm: break
        G_undirected.remove_nodes_from(to_rm)

    lcc = max(nx.connected_components(G_undirected), key=len)
    return G_undirected.subgraph(lcc).copy()


def build_lehigh_graph():
    point, dist = (40.6053, -75.3780), 800
    G = ox.graph_from_point(point, dist=dist, network_type='drive_service')
    try:
        G_undirected = ox.utils_graph.get_undirected(G)
    except AttributeError:
        G_undirected = ox.convert.to_undirected(G)

    u_cut, v_cut, u_del = 259302817, 478228138, 1016142119
    if G_undirected.has_edge(u_cut, v_cut):
        G_copy = G_undirected.copy(); G_copy.remove_edge(u_cut, v_cut)
        G_undirected.remove_nodes_from(list(nx.node_connected_component(G_copy, v_cut)))
    if u_del in G_undirected: G_undirected.remove_node(u_del)
    if G_undirected.has_edge(6204650796, 6204650804):
        G_undirected.remove_edge(6204650796, 6204650804)
    for n in {259317685}: 
        if n in G_undirected: G_undirected.remove_node(n)
    G_undirected.remove_nodes_from([n for n in {248551974,1016142066,1016377132,1016376957,1016377134} if n in G_undirected])
    G_undirected.remove_nodes_from([n for n in {259368711,472425550,259257425,1121706772,1121706745,258574525,259274110} if n in G_undirected])
    for u_e,v_e in [(335361039,335360953),(1016142113,259346266),(60978330,60978331)]:
        if u_e in G_undirected and v_e in G_undirected:
            x1,y1 = G_undirected.nodes[u_e]['x'],G_undirected.nodes[u_e]['y']
            x2,y2 = G_undirected.nodes[v_e]['x'],G_undirected.nodes[v_e]['y']
            G_undirected.add_edge(u_e,v_e,key=0,length=ox.distance.great_circle(y1,x1,y2,x2),
                                  geometry=LineString([(x1,y1),(x2,y2)]))
    nodes_116 = [n for n in G_undirected.nodes() if str(n).startswith("1166459")]
    for n in nodes_116:
        if G_undirected.degree(n) == 1: G_undirected.remove_node(n)
    for n in [11664593244,11664593242,11664593245,11664593249]:
        if n in G_undirected: G_undirected.remove_node(n)
    while True:
        rm = [nd for nd,d in G_undirected.degree() if d < 2]
        if not rm: break
        G_undirected.remove_nodes_from(rm)
    lcc = max(nx.connected_components(G_undirected), key=len)
    return G_undirected.subgraph(lcc).copy()


# ==============================
# 2) Setup
# ==============================
if USE_LEHIGH:
    print("Building Lehigh University graph...")
    G_undirected = build_lehigh_graph()
    MAP_NAME = "Lehigh"
else:
    print("Building ASU Polytechnic Campus graph...")
    G_undirected = build_asu_graph()
    MAP_NAME = "ASU"

n, _ = G_undirected.number_of_nodes(), G_undirected.number_of_edges()
nodes = sorted(G_undirected.nodes())
node_to_idx = {v: i for i, v in enumerate(nodes)}

edge_list = sorted([tuple(sorted((u,v))) for u,v in G_undirected.edges()])
edge_list = list(dict.fromkeys(edge_list))
edge_to_idx = {e: i for i, e in enumerate(edge_list)}
m = len(edge_list)

basis_meta, _ = find_horton_gram_mcb(G_undirected, edge_list)
k = len(basis_meta)
C_matrix = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    for i in range(m):
        if (c_data["bv"] >> i) & 1: C_matrix[i, j] = 1
C_oriented = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    cycle_edges = [edge_list[i] for i in range(m) if C_matrix[i,j]==1]
    cs = nx.Graph(cycle_edges)
    try:
        cn = [e[0] for e in nx.find_cycle(cs)]
    except Exception:
        cn = list(cs.nodes())
    for i in range(len(cn)):
        u_c,v_c = cn[i], cn[(i+1)%len(cn)]
        u,v = sorted((u_c,v_c))
        if (u,v) in edge_to_idx:
            idx = edge_to_idx[(u,v)]
            C_oriented[idx,j] = +1 if (u_c==u and v_c==v) else -1

SPEED = 16
tau = np.zeros(m)
for i,(u,v) in enumerate(edge_list):
    ed = G_undirected[u][v]
    d = ed[0] if 0 in ed else next(iter(ed.values()))
    tau[i] = round(d.get('length',100.0)/SPEED, 1)

T_max = int(sum(tau)*1.5)
np.random.seed(0)
r = np.random.randint(1, 11, n)
V_MIN_vec = np.random.randint(1, 2, n)

A_abs = np.zeros((n, m), dtype=int)
for j,(u,v) in enumerate(edge_list):
    A_abs[node_to_idx[u],j] = 1; A_abs[node_to_idx[v],j] = 1

n_vars = 2*k+m+m; M_M = 20000
c_obj_B = np.zeros(n_vars)
c_obj_B[2*k:2*k+m] = 0.5*A_abs.T@r
A1=np.hstack([-C_oriented, C_oriented,np.eye(m),np.zeros((m,m))])
A2=np.hstack([ C_oriented,-C_oriented,np.eye(m),np.zeros((m,m))])
A3=np.hstack([-C_oriented, C_oriented,np.eye(m),M_M*np.eye(m)])
A4=np.hstack([ C_oriented,-C_oriented,np.eye(m),-M_M*np.eye(m)])
A_t=np.zeros((1,n_vars)); A_t[0,2*k:2*k+m]=tau
A_nd=np.zeros((n,n_vars)); A_nd[:,2*k:2*k+m]=A_abs
A_ub_B=np.vstack([-A1,-A2,A3,A4,A_t,-A_nd])
lb_B=np.zeros(n_vars); ub_B=np.full(n_vars,np.inf); ub_B[2*k+m:]=1
b_ub=np.hstack([np.zeros(m),np.zeros(m),M_M*np.ones(m),np.zeros(m),[T_max],-2*V_MIN_vec])

print("\nSolving MILP (flow_B)...")
_t0=time.time()
res_B=milp(c=-c_obj_B,constraints=LinearConstraint(A_ub_B,-np.inf,b_ub),
           bounds=Bounds(lb_B,ub_B),integrality=np.ones(n_vars,dtype=int))
milp_time=time.time()-_t0
if not res_B.success: raise RuntimeError("MILP failed")
sol=np.round(res_B.x).astype(int)
w_signed=sol[:k]-sol[k:2*k]
z_opt=sol[2*k:2*k+m]
flow_B=C_oriented@w_signed
v_opt_B=0.5*A_abs@z_opt
J_B=float(r@v_opt_B)
print(f"  Reward J_B={J_B:.2f},  runtime={milp_time:.3f}s")

# ==============================
# 3) Select blocked edges
# ==============================
active_idx=[i for i in range(m) if flow_B[i]!=0]
rng=np.random.default_rng(seed=7)
shuffled=rng.permutation(active_idx).tolist()
blocked_idx1=shuffled[0]; u1,v1=edge_list[blocked_idx1]
candidates2=[i for i in shuffled[1:] if u1 not in edge_list[i] and v1 not in edge_list[i]]
if not candidates2: raise RuntimeError("No non-adjacent pair.")
blocked_idx2=candidates2[0]
blocked_edge1=edge_list[blocked_idx1]; blocked_flow1=int(flow_B[blocked_idx1])
blocked_edge2=edge_list[blocked_idx2]; blocked_flow2=int(flow_B[blocked_idx2])
d1,d2=-blocked_flow1,-blocked_flow2
print(f"Blocked: {blocked_edge1} (flow={blocked_flow1}),  {blocked_edge2} (flow={blocked_flow2})")

def _check_eulerian(fv):
    DG_t=nx.MultiDiGraph(); DG_t.add_nodes_from(nodes)
    for i,(u,v) in enumerate(edge_list):
        f=int(fv[i])
        if f>0:
            for _ in range(f): DG_t.add_edge(u,v)
        elif f<0:
            for _ in range(abs(f)): DG_t.add_edge(v,u)
    in_d=dict(DG_t.in_degree()); out_d=dict(DG_t.out_degree())
    ni=[v for v in nodes if in_d[v]+out_d[v]>0]
    bal=all(in_d[v]==out_d[v] for v in ni) if ni else True
    wc=nx.is_weakly_connected(DG_t.subgraph(ni)) if ni else True
    return bal and wc, bal, wc

T_orig_val = float(tau @ np.abs(flow_B))

# ==============================
# 4) METHOD B: MIQP Cycle-Basis Repair
# ==============================
print("\n"+"="*60)
print(f"METHOD B: MIQP Cycle-Basis Repair — {MAP_NAME}")
print("="*60)

DELTA_T_TOL = 0.2
beta_var = cp.Variable(k, integer=True)
Cm_expr  = C_oriented @ beta_var
f_expr   = flow_B + Cm_expr
z_abs    = cp.Variable(m, integer=True)
b_sign   = cp.Variable(m, boolean=True)
BIG_M    = 20000
constr = [
    Cm_expr[blocked_idx1] == d1,
    Cm_expr[blocked_idx2] == d2,
    z_abs >= f_expr, z_abs >= -f_expr,
    z_abs <= f_expr + BIG_M*(1-b_sign),
    z_abs <= -f_expr + BIG_M*b_sign,
]
T_new   = tau @ z_abs
delta_T = T_new - T_orig_val
eta_var = cp.Variable()
constr += [eta_var >= delta_T, eta_var >= -delta_T, eta_var <= DELTA_T_TOL]
prob = cp.Problem(cp.Minimize(cp.sum_squares(Cm_expr)), constr)

_t0_miqp = time.time()
prob.solve(solver=cp.SCIP, verbose=False)
miqp_time = time.time() - _t0_miqp

if prob.status in ("optimal","optimal_inaccurate"):
    beta_star = np.round(beta_var.value).astype(int)
    Cm = C_oriented @ beta_star
    flow_corrected = flow_B + Cm
    edge_changes = int(np.count_nonzero(Cm))
    node_changes = int(np.sum((A_abs @ np.abs(Cm)) > 0))
    dT = float(tau @ np.abs(flow_corrected)) - T_orig_val
    inner_prod = int(flow_B @ Cm)
    euler_ok, _, _ = _check_eulerian(flow_corrected)
else:
    print(f"  MIQP failed: status={prob.status}")
    Cm=np.zeros(m,dtype=int); flow_corrected=flow_B.copy()
    edge_changes=node_changes=0; dT=0.0; inner_prod=0; euler_ok=False

v1_ok = flow_corrected[blocked_idx1] == 0
v2_ok = flow_corrected[blocked_idx2] == 0

print()
print("="*70)
print(f"  Map                  : {MAP_NAME}")
print(f"  MILP (flow_B) reward : {J_B:.2f}")
print(f"  MILP (flow_B) time   : {milp_time:.3f}s")
print(f"  MIQP repair time     : {miqp_time:.3f}s")
print(f"  Eulerian             : {euler_ok}")
print(f"  Edge changes         : {edge_changes}")
print(f"  Node changes         : {node_changes}")
print(f"  ΔT                   : {dT:+.1f}s")
print(f"  ⟨flow_B, Cm⟩         : {inner_prod}")
print(f"  ||Cm||_2²            : {int(Cm@Cm)}")
print(f"  ||Cm||_1             : {int(np.sum(np.abs(Cm)))}")
print(f"  Blocked edge 1 zeroed: {v1_ok}  (flow_corrected[{blocked_idx1}]={flow_corrected[blocked_idx1]})")
print(f"  Blocked edge 2 zeroed: {v2_ok}  (flow_corrected[{blocked_idx2}]={flow_corrected[blocked_idx2]})")
print("="*70)
