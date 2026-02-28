import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from matplotlib.patches import FancyArrowPatch

# === Global Setup ===
print("="*70)
print("TOY MODEL VISUALIZATION: Asymmetric Cost & Cancellation")
print("="*70)

# Graph Definition
n, m, k = 4, 5, 2
nodes = [0, 1, 2, 3]
pos = {0: (0, 0), 1: (0.5, 0.866), 2: (1, 0), 3: (0.5, -0.866)}
edge_list = [(0,1), (0,2), (0,3), (1,2), (2,3)]
# C_oriented
C_oriented = np.array([
    [+1,  0],  # e0 (0,1): Cycle1 (+)
    [-1, +1],  # e1 (0,2): Cycle1 (-), Cycle2 (+)  <-- Shared Edge
    [ 0, -1],  # e2 (0,3): Cycle2 (-)
    [+1,  0],  # e3 (1,2): Cycle1 (+)
    [ 0, +1],  # e4 (2,3): Cycle2 (+)
], dtype=int)
# Incidence Matrix
A_abs = np.array([
    [1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1],
], dtype=int)

# === Asymmetric Scenario (Scene 3) ===
tau3 = np.array([5.0, 2.0, 1.0, 5.0, 1.0])
r3 = np.array([1.0, 1.0, 5.0, 5.0])
T_max3 = 3000
M = 10000
n_vars = 2*k + m + m  # w+, w-, z, b

# Constraints
c_obj3 = np.zeros(n_vars)
c_obj3[2*k:2*k+m] = 0.5 * (A_abs.T @ r3)
A1 = np.hstack([C_oriented, -C_oriented, -np.eye(m), np.zeros((m, m))])
A2 = np.hstack([-C_oriented, C_oriented, -np.eye(m), np.zeros((m, m))])
A3 = np.hstack([-C_oriented, C_oriented, np.eye(m), M * np.eye(m)])
A4 = np.hstack([C_oriented, -C_oriented, np.eye(m), -M * np.eye(m)])
A_time = np.zeros((1, n_vars)); A_time[0, 2*k:2*k+m] = tau3
A_ub = np.vstack([A1, A2, A3, A4, A_time])
b_ub = np.hstack([np.zeros(m), np.zeros(m), M*np.ones(m), np.zeros(m), [T_max3]])

# Force w+[0] >= 100
lb = np.zeros(n_vars); lb[0] = 100
ub = np.full(n_vars, np.inf); ub[2] = 0 # w-[0]=0
ub[2*k+m:] = 1

# Solve
res = milp(c=-c_obj3, constraints=LinearConstraint(A_ub, -np.inf, b_ub),
           bounds=Bounds(lb=lb, ub=ub), integrality=np.ones(n_vars, dtype=int))

if not res.success:
    print("Optimization Failed!")
    exit()

sol = np.round(res.x).astype(int)
w_signed = sol[:k] - sol[k:2*k]
z_opt = sol[2*k:2*k+m]
flow = C_oriented @ w_signed
v_opt = 0.5 * A_abs @ z_opt

print(f"w_signed = {w_signed}")
print(f"flow     = {flow}")
print(f"z_opt    = {z_opt}")

# === Visualize ===
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# 1. Draw Cycle Directions (Basis Definition)
# Cycle 1: 0->1->2->0 (Red dashed)
c1_path = [(0,1), (1,2), (2,0)]
offset = 0.05
for u, v in c1_path:
    p1 = np.array(pos[u])
    p2 = np.array(pos[v])
    # Offset slightly outward
    center = np.mean([pos[0], pos[1], pos[2]], axis=0)
    d1 = p1 - center; d1 = d1 / np.linalg.norm(d1) * offset
    d2 = p2 - center; d2 = d2 / np.linalg.norm(d2) * offset
    
    arrow = FancyArrowPatch(posA=(p1+d1), posB=(p2+d2), 
                            arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                            color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                            mutation_scale=15)
    # ax.add_patch(arrow) # Too cluttered, maybe just key

# 2. Draw Edges based on Flow
for i, (u, v) in enumerate(edge_list):
    f = flow[i]
    z = z_opt[i]
    
    p_u, p_v = np.array(pos[u]), np.array(pos[v])
    
    # Color based on flow
    if f > 0:
        col = 'green'  # u->v
        start, end = p_u, p_v
    elif f < 0:
        col = 'blue'   # v->u
        start, end = p_v, p_u
    else:
        col = 'gray'   # Cancellation or 0 flow
        start, end = p_u, p_v
    
    if z == 0:
        # CANCELLATION! Draw dashed gray line
        ax.plot([p_u[0], p_v[0]], [p_u[1], p_v[1]], 
                color='gray', linestyle=':', linewidth=1, alpha=0.5)
        mx, my = (p_u + p_v)/2
        ax.text(mx, my, "CANCELLED\n(z=0)", color='gray', fontsize=8, ha='center', va='center')
        continue

    # Draw Arrow for Flow
    width = 1 + z / 80
    arrow = FancyArrowPatch(posA=start, posB=end,
                            arrowstyle='-|>', 
                            connectionstyle='arc3,rad=0',
                            color=col, linewidth=width, alpha=0.9,
                            mutation_scale=20)
    ax.add_patch(arrow)
    
    # Label amount
    mx, my = (p_u + p_v)/2
    ax.text(mx, my, f"flow={abs(f)}\nz={z}", fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 3. Draw Nodes
for v in nodes:
    ax.scatter(*pos[v], s=600, c='lightyellow', edgecolors='black', zorder=10)
    ax.text(*pos[v], f"Node {v}\nvisit={v_opt[v]:.0f}", ha='center', va='center', fontsize=9)

# Legend for Cycles
# Draw dummy lines for legend
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='green', lw=2, label='Pos Flow (u->v)'),
    Line2D([0], [0], color='blue', lw=2, label='Neg Flow (v->u)'),
    Line2D([0], [0], color='gray', linestyle=':', label='Cancelled (z=0)'),
    Line2D([0], [0], color='w', label=f"Cycle 1 (Top): {w_signed[0]} (+CW)"),
    Line2D([0], [0], color='w', label=f"Cycle 2 (Bot): {w_signed[1]} (-CCW)")
]
ax.legend(handles=custom_lines, loc='upper right')

ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Asymmetric Flow & Cancellation Visualization", fontsize=14)

plt.tight_layout()
plt.savefig("toy_model_asymmetric.png", dpi=150)
print("Saved visualization to toy_model_asymmetric.png")
plt.close()

# === Visualize 2: Graph Basis & Reference Directions ===
print("\nGenerating Basis Visualization...")
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))

# Draw Reference Edge Directions (gray arrows)
for i, (u, v) in enumerate(edge_list):
    p_u, p_v = np.array(pos[u]), np.array(pos[v])
    
    # Reference Direction u->v
    arrow = FancyArrowPatch(posA=p_u, posB=p_v,
                            arrowstyle='-|>', 
                            color='gray', linewidth=2, alpha=0.3,
                            mutation_scale=20, label='Ref Edge')
    ax2.add_patch(arrow)
    
    mx, my = (p_u + p_v)/2
    ax2.text(mx, my, f"e{i}\n({u}->{v})", color='gray', fontsize=8, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Draw Cycle 1 Basis (Red dashed)
c1_edges = [(0,1), (1,2), (2,0)]
offset = 0.08
center_c1 = np.mean([pos[0], pos[1], pos[2]], axis=0)

for u, v in c1_edges:
    p1 = np.array(pos[u])
    p2 = np.array(pos[v])
    
    # Offset inward towards cycle center
    d1 = center_c1 - p1; d1 = d1 / np.linalg.norm(d1) * offset
    d2 = center_c1 - p2; d2 = d2 / np.linalg.norm(d2) * offset
    
    # Interpret direction relative to reference
    ref_u, ref_v = sorted((u,v))
    idx = edge_list.index((ref_u, ref_v))
    sign = C_oriented[idx, 0] # Cycle 1 col
    
    label_txt = f"{'+1' if sign > 0 else '-1'}"
    col = 'red'
    
    arrow = FancyArrowPatch(posA=(p1+d1), posB=(p2+d2), 
                            arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                            color=col, linestyle='--', alpha=0.8, linewidth=1.5,
                            mutation_scale=15)
    ax2.add_patch(arrow)
    
    # Mark sign
    mx, my = (p1+d1 + p2+d2)/2
    ax2.text(mx, my, label_txt, color=col, fontweight='bold', fontsize=10)

# Draw Cycle 2 Basis (Blue dashed)
c2_edges = [(0,2), (2,3), (3,0)]
center_c2 = np.mean([pos[0], pos[2], pos[3]], axis=0)

for u, v in c2_edges:
    p1 = np.array(pos[u])
    p2 = np.array(pos[v])
    
    # Offset inward
    d1 = center_c2 - p1; d1 = d1 / np.linalg.norm(d1) * offset
    d2 = center_c2 - p2; d2 = d2 / np.linalg.norm(d2) * offset
    
    ref_u, ref_v = sorted((u,v))
    idx = edge_list.index((ref_u, ref_v))
    sign = C_oriented[idx, 1] # Cycle 2 col

    label_txt = f"{'+1' if sign > 0 else '-1'}"
    col = 'blue'

    arrow = FancyArrowPatch(posA=(p1+d1), posB=(p2+d2), 
                            arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                            color=col, linestyle='--', alpha=0.8, linewidth=1.5,
                            mutation_scale=15)
    ax2.add_patch(arrow)
    mx, my = (p1+d1 + p2+d2)/2
    ax2.text(mx, my, label_txt, color=col, fontweight='bold', fontsize=10)

# Nodes
for v in nodes:
    ax2.scatter(*pos[v], s=600, c='white', edgecolors='black', zorder=10)
    ax2.text(*pos[v], str(v), ha='center', va='center', fontsize=12, fontweight='bold')

# Legend
custom_lines = [
    Line2D([0], [0], color='gray', lw=2, alpha=0.3, label='Reference Edge (u->v)'),
    Line2D([0], [0], color='red', linestyle='--', lw=2, label='Cycle 1 Basis (+CW)'),
    Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Cycle 2 Basis (+CCW)'),
]
ax2.legend(handles=custom_lines, loc='upper right')
ax2.set_title("Graph Basis & Reference Directions", fontsize=14)
ax2.axis('off')

plt.tight_layout()
plt.savefig("toy_model_basis.png", dpi=150)
print("Saved visualization to toy_model_basis.png")
