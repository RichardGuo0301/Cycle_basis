"""
Paper Fig 1 Reproduction Demo
Visualizes the Physical Layer from the paper (13 nodes, 19 edges) with manual layout.
Computes Cycle Basis and Dual Graph.
"""

import matplotlib
matplotlib.use('Agg')

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Ensure numpy compatibility
if not hasattr(np, 'alltrue'):
    np.alltrue = np.all

from cycle_basis_demo import get_cycle_basis_info, get_oriented_cycle_vectors
from clean_cycle_basis import find_cleanest_cycle_basis
from orthogonal_cycle_basis import find_orthogonal_cycle_basis
from cycle_dual_graph_demo import (
    build_cycle_dual_graph, 
    visualize_dual_comparison, 
    visualize_individual_cycles,
    visualize_incidence_matrix,
    get_oriented_incidence_matrix,
    get_cycle_edges
)

def create_fig1_graph():
    """
    Creates the graph from Fig 1 using user-provided layout and edges.
    """
    G = nx.Graph()
    
    # 1. Define nodes and coordinates (Manual layout)
    pos = {
        1:  (0, 5),    # Top Left
        4:  (0.5, 1),  # Bottom Left
        
        2:  (3.5, 6),  # Upper layer
        3:  (3, 3.5),  # Middle layer left
        5:  (3.5, 0),  # Lower layer
        
        6:  (5.5, 3),  # Kernel center
        
        7:  (7.5, 6),  # Upper right
        8:  (7.5, 3.5),# Middle right
        10: (8, -0.5), # Lower right
        
        9:  (9.5, 3),  # Convergence point
        
        11: (11.5, 4), # Branch Top
        12: (12, 1),   # Branch Bottom
        
        13: (14, 5)    # Sink
    }
    
    G.add_nodes_from(pos.keys())
    nx.set_node_attributes(G, pos, 'pos')

    # 2. Define edges (Undirected for structure)
    edges = [
        (1, 2), (1, 3), (1, 4),   # e1, e2, e3
        (2, 3), (2, 6), (2, 7),   # e4, e5, e6
        (3, 6),                   # e7
        (4, 5),                   # e8
        (5, 6), (5, 10),          # e9, e10
        (7, 8), (7, 9),           # e11, e13
        (6, 8),                   # e12
        (8, 9),                   # e14
        (10, 9),                  # e15 (9-10)
        (9, 11), (9, 12),         # e16, e17
        (11, 13),                 # e18
        (12, 13)                  # e19
    ]
    G.add_edges_from(edges)
    
    # Create Directed Version for Visualization Arrows
    # (Orientation similar to paper flow: Left -> Right roughly)
    G_directed = nx.DiGraph()
    G_directed.add_nodes_from(G.nodes())
    nx.set_node_attributes(G_directed, pos, 'pos')
    
    for u, v in edges:
        # Heuristic direction: from left (smaller x) to right (larger x)
        # Or just smaller ID to larger ID? Let's use X coordinate.
        x_u = pos[u][0]
        x_v = pos[v][0]
        if x_u < x_v:
            G_directed.add_edge(u, v)
        elif x_u > x_v:
            G_directed.add_edge(v, u)
        else:
            # Vertical edges: Top down?
            y_u = pos[u][1]
            if y_u > pos[v][1]:
                G_directed.add_edge(u, v)
            else:
                G_directed.add_edge(v, u)

    return G_directed, G

def main():
    print("="*60)
    print("Paper Fig 1 Reproduction - Cycle Analysis")
    print("="*60)
    
    # 1. Create Graph
    G_directed, G_undirected = create_fig1_graph()
    num_nodes = G_directed.number_of_nodes()
    num_edges = G_directed.number_of_edges()
    print(f"Graph Created: {num_nodes} nodes, {num_edges} edges")
    
    # Verify Cycle Count: m - n + 1
    # 19 - 13 + 1 = 7. Correct.
    expected_cycles = num_edges - num_nodes + 1
    print(f"Expected Cycle Basis Size: {expected_cycles}")
    
    # 2. Compute Basis (using ORTHOGONAL NULL SPACE BASIS)
    # This selects cycles that are most orthogonal in the null space
    cycles = find_orthogonal_cycle_basis(G_undirected)
    print(f"\nOrthogonal Cycle Basis Size: {len(cycles)}")
    
    # Reconstruct metadata
    edges_list = []
    
    from cycle_dual_graph_demo import get_cycle_edges
    
    for c in cycles:
        edges_set = get_cycle_edges(G_undirected, c)
        edges_list.append(edges_set)
        print(f"  Cycle: {sorted(list(c))}")
    
    oriented_vectors = get_oriented_cycle_vectors(G_directed, cycles)
        
    # 3. Build Dual Graph
    DualG = build_cycle_dual_graph(G_undirected, cycles)
    print(f"Dual Graph: {DualG.number_of_nodes()} nodes, {DualG.number_of_edges()} edges")
    
    # 4. Visualization
    # Colors suitable for ~7 cycles
    cmap = plt.cm.get_cmap('tab10', len(cycles))
    cycle_colors = [mcolors.to_hex(cmap(i)) for i in range(len(cycles))]
    
    # Visualize Individual Cycles
    visualize_individual_cycles(
        G_directed, G_undirected, cycles, edges_list, 
        oriented_vectors, 
        save_path="fig1_cycles.png",
        custom_colors=cycle_colors
    )
    
    # Visualize Dual Graph Comparison
    visualize_dual_comparison(
        G_directed, G_undirected, cycles, DualG,
        save_path="fig1_dual_graph.png",
        custom_colors=cycle_colors
    )
    
    # Visualize Incidence Matrix
    I_o, nodes, edges = get_oriented_incidence_matrix(G_directed)
    visualize_incidence_matrix(I_o, nodes, edges, save_path="fig1_incidence_matrix.png")

if __name__ == "__main__":
    main()
