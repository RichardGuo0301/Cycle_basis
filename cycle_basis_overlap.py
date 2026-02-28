"""
Cycle Basis Overlap Visualization

Left panel : Original graph — each edge labelled with which cycle bases use it;
             color intensity = how many bases share it.
Right panel: Abstract overlap graph — each node is one cycle basis (Cj),
             an edge Cj1–Cj2 means they share ≥1 original graph edge;
             edge label = number of shared original edges.

Uses same graph seeds as edge_block_2MIQP.py  (np.random.seed(42), graph seed=100).
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from cycle_basis_demo import create_random_polygon_graph
from horton_gram_mcb import find_horton_gram_mcb

# ── same setup as edge_block_2MIQP.py ───────────────────────────────────────
np.random.seed(42)
G_directed, G_undirected = create_random_polygon_graph(12, 10, seed=100)
n, m = G_undirected.number_of_nodes(), G_undirected.number_of_edges()
nodes = sorted(G_undirected.nodes())

edge_list = sorted([tuple(sorted(e)) for e in G_undirected.edges()])
edge_to_idx = {e: i for i, e in enumerate(edge_list)}

basis_meta, _ = find_horton_gram_mcb(G_undirected, edge_list)
k = len(basis_meta)

# C_matrix[i, j] = 1  iff  edge i is in cycle basis j
C_matrix = np.zeros((m, k), dtype=int)
for j, c_data in enumerate(basis_meta):
    for i in range(m):
        if (c_data["bv"] >> i) & 1:
            C_matrix[i, j] = 1

# edge_to_cycles[i] → list of cycle basis indices that contain edge i
edge_to_cycles = {i: [j for j in range(k) if C_matrix[i, j] == 1]
                  for i in range(m)}

# ── Build cycle basis overlap graph ─────────────────────────────────────────
overlap_G = nx.Graph()
overlap_G.add_nodes_from(range(k))
shared_edges_dict = {}   # (j1, j2) → [edge_indices, ...]

for j1 in range(k):
    for j2 in range(j1 + 1, k):
        shared = [i for i in range(m)
                  if C_matrix[i, j1] == 1 and C_matrix[i, j2] == 1]
        if shared:
            overlap_G.add_edge(j1, j2, count=len(shared))
            shared_edges_dict[(j1, j2)] = shared

# ── Console summary ──────────────────────────────────────────────────────────
print(f"\nGraph: n={n}, m={m}, k={k}")
print(f"\n{'Edge':>5}  {'(u,v)':>12}  {'In cycle bases':>30}  #bases")
print("-" * 60)
for i, (u, v) in enumerate(edge_list):
    cycles = edge_to_cycles[i]
    label = "C" + ", C".join(str(j) for j in cycles) if cycles else "—"
    print(f"  {i:>2}  {str((u,v)):>12}  {label:>30}  {len(cycles)}")

print(f"\nCycle basis overlap ({overlap_G.number_of_edges()} sharing pairs):")
for (j1, j2), se in sorted(shared_edges_dict.items()):
    edges_str = "  ".join(str(edge_list[i]) for i in se)
    print(f"  C{j1} ↔ C{j2}: {len(se)} edge(s) → {edges_str}")

isolated = [j for j in range(k) if overlap_G.degree(j) == 0]
if isolated:
    print(f"\n  Isolated bases (share no edge with anyone): {['C'+str(j) for j in isolated]}")

# ── Figure: two panels ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))

pos = (nx.get_node_attributes(G_undirected, "pos")
       or nx.spring_layout(G_undirected, seed=42))

cmap_heat = plt.cm.YlOrRd
max_usage = max(len(v) for v in edge_to_cycles.values())

# ── Panel 1: original graph ──────────────────────────────────────────────────
# ghost all edges first
nx.draw_networkx_edges(G_undirected, pos, width=0.8, alpha=0.12,
                       edge_color="gray", ax=ax1)

# color each edge by usage count (convert RGBA→hex to avoid numpy2 bug in networkx)
for i, (u, v) in enumerate(edge_list):
    cnt = len(edge_to_cycles[i])
    color_hex = mcolors.to_hex(cmap_heat(cnt / max(max_usage, 1)))
    lw = 1.5 + cnt * 1.5
    nx.draw_networkx_edges(G_undirected, pos, edgelist=[(u, v)],
                           width=lw, edge_color=color_hex, alpha=0.92, ax=ax1)

nx.draw_networkx_nodes(G_undirected, pos, node_size=650,
                       node_color="lightsteelblue",
                       edgecolors="black", linewidths=2.0, ax=ax1)
nx.draw_networkx_labels(G_undirected, pos, font_size=10,
                        font_weight="bold", ax=ax1)

# edge labels: list of cycle bases
edge_label_dict = {}
for i, (u, v) in enumerate(edge_list):
    cycles = edge_to_cycles[i]
    if cycles:
        edge_label_dict[(u, v)] = "C" + ",".join(str(j) for j in cycles)

nx.draw_networkx_edge_labels(
    G_undirected, pos, edge_label_dict, font_size=7, ax=ax1,
    label_pos=0.38,
    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7)
)

ax1.set_title(
    f"Original Graph — Edge–Cycle Basis Membership\n"
    f"n={n}  m={m}  k={k}  |  Label = which bases use the edge  |  "
    f"Darker/thicker = more bases share it",
    fontsize=12, fontweight="bold"
)
ax1.axis("off")

from matplotlib.patches import Patch
legend_els = [Patch(facecolor=cmap_heat(c / max(max_usage, 1)),
                    edgecolor="black",
                    label=f"shared by {c} basis")
              for c in range(1, max_usage + 1)]
ax1.legend(handles=legend_els, loc="lower left", fontsize=9,
           title="Edge usage", title_fontsize=9)

# ── Panel 2: cycle basis overlap graph ──────────────────────────────────────
ov_pos = nx.spring_layout(overlap_G, seed=42, k=2.8)

node_degs = {j: overlap_G.degree(j) for j in range(k)}
max_deg = max(node_degs.values()) if node_degs else 1
node_sizes  = [900 + basis_meta[j]["length"] * 90 for j in range(k)]
node_colors = [plt.cm.Blues(0.25 + 0.65 * node_degs[j] / max(max_deg, 1))
               for j in range(k)]

nx.draw_networkx_nodes(overlap_G, ov_pos, node_size=node_sizes,
                       node_color=node_colors,
                       edgecolors="black", linewidths=2.0, ax=ax2)

node_labels = {j: f"C{j}\nlen={basis_meta[j]['length']}\ndeg={node_degs[j]}"
               for j in range(k)}
nx.draw_networkx_labels(overlap_G, ov_pos, node_labels,
                        font_size=7.5, font_weight="bold", ax=ax2)

if overlap_G.number_of_edges() > 0:
    edge_widths = [overlap_G[u][v]["count"] * 1.8
                   for u, v in overlap_G.edges()]
    nx.draw_networkx_edges(overlap_G, ov_pos, width=edge_widths,
                           edge_color="steelblue", alpha=0.65, ax=ax2)
    edge_labels_ov = {(j1, j2): str(overlap_G[j1][j2]["count"])
                      for j1, j2 in overlap_G.edges()}
    nx.draw_networkx_edge_labels(
        overlap_G, ov_pos, edge_labels_ov, font_size=11, ax=ax2,
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.85)
    )

ax2.set_title(
    f"Cycle Basis Overlap Graph\n"
    f"{k} nodes (one per cycle basis)  |  "
    f"{overlap_G.number_of_edges()} sharing pairs  |  "
    f"Edge label = # shared original edges  |  Node size ∝ cycle length",
    fontsize=12, fontweight="bold"
)
ax2.axis("off")

plt.tight_layout()
os.makedirs("visuals", exist_ok=True)
out_path = "visuals/cycle_basis_overlap.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")
