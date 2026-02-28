"""
IEEE 30 Bus System - Clean Cycle Basis Demo
Uses precise user layout and optimized Geometric Cycle Basis.
No incidence matrix output.
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

from cycle_basis_demo import get_oriented_cycle_vectors
from horton_gram_mcb import find_horton_gram_mcb  # Use new algorithm
from cycle_dual_graph_demo import (
    build_cycle_dual_graph, 
    visualize_dual_comparison, 
    visualize_individual_cycles,
    get_cycle_edges 
)

def create_ieee30_graph_clean():
    """
    Creates the IEEE 30 Bus System graph with manual layout matching user's diagram.
    """
    G_directed = nx.DiGraph()
    
    # 1. Define nodes and coordinates (matches user provided layout)
    pos = {
        1:  (0, 0),
        2:  (2, 0),
        3:  (0.2, 2),    # Slightly right or vertical
        4:  (2, 4),
        5:  (4, 2.5),
        6:  (6, 0),
        7:  (4.5, 1.2),  # Between 5 and 6
        8:  (10, 0),
        9:  (7.5, 2.5),
        10: (6, 4.5),
        11: (9, 2),
        12: (2, 6),
        13: (0.5, 4),    # Left of 4
        14: (0.5, 5.5),  # Lower left of 12
        15: (1, 8),
        16: (3.5, 5),
        17: (4.5, 4.5),
        18: (3, 7.5),
        19: (4.5, 6.8),
        20: (6, 6),
        21: (5.5, 7.5),
        22: (7.5, 5.5),
        23: (4, 9),      # Right of 15, Left of 24
        24: (7, 8),
        25: (7, 9),
        26: (5.5, 9),
        27: (6, 10.5),
        28: (10, 10.5),
        29: (4, 11),     # Left of 27
        30: (4, 10)      # Below 29
    }
    
    G_directed.add_nodes_from(pos.keys())
    nx.set_node_attributes(G_directed, pos, 'pos')

    # 2. Define edges (Based on user's directed edge list)
    edges = [
        (1, 2), (1, 3),
        (2, 4), (2, 5), (2, 6),
        (3, 4),
        (4,5), (4, 12), (4, 13), 
        (5, 7), (5, 6),
        (6, 7), (6, 8), (6, 9), (6, 10), (6,11),
        (8, 28),
        (9, 10), (9, 11),
        (10, 17), (10, 20), (10, 22),
        (12, 14), (12, 15), (12, 16),
        (14, 15),
        (15, 18), (15, 23),
        (16, 17),
        (18, 19),
        (19, 20),
        (20, 21),
        (22, 24),
        (23, 24),
        (24, 25),
        (25, 26), (25, 27),
        (27, 28), (27, 29), (27, 30),
        (29, 30)
    ]
    G_directed.add_edges_from(edges)
    
    # Create undirected version for basis calculation
    G_undirected = G_directed.to_undirected()
    
    return G_directed, G_undirected

def main():
    print("="*60)
    print("IEEE 30 Bus System - Clean Geometric Cycle Analysis")
    print("="*60)
    
    # 1. Create Graph
    G_directed, G_undirected = create_ieee30_graph_clean()
    num_nodes = G_directed.number_of_nodes()
    num_edges = G_directed.number_of_edges()
    print(f"Graph Created: {num_nodes} nodes, {num_edges} edges")
    
    # Verify Cycle Count: m - n + 1 (assuming 1 connected component)
    expected_cycles = num_edges - num_nodes + 1
    print(f"Expected Cycle Basis Size: {expected_cycles}")
    
    # 2. Compute Basis using Horton-Gram MCB selection
    print("Computing Optimized Horton-Gram Cycle Basis...")
    basis_meta, _ = find_horton_gram_mcb(G_undirected)
    print(f"Computed Solution Size: {len(basis_meta)}")
    
    # Reconstruct cycles from basis_meta
    cycles = []
    edges_list = []
    
    for i, c_data in enumerate(basis_meta):
        # Explicit edges from the algorithm
        es_list = c_data['edges']
        es_set = set(es_list)
        edges_list.append(es_set) # For visualizer
        
        # Reconstruct ordered nodes for visualization
        H = nx.Graph()
        H.add_edges_from(es_list)
        try:
            cycles_sub = nx.cycle_basis(H)
            if cycles_sub:
                c_nodes = max(cycles_sub, key=len)
            else:
                c_nodes = [] # Should not happen
        except:
             c_nodes = []
        
        cycles.append(c_nodes)
        
        print(f"C{i+1}: Length {c_data['length']}, Nodes: {sorted(list(c_nodes))}")
        
    oriented_vectors = get_oriented_cycle_vectors(G_directed, cycles)

    # Prepare data for tables
    edge_list = sorted(G_undirected.edges())
    # Canonical edge ordering: if (u,v) in list, ensure u<v
    edge_list = [tuple(sorted(e)) for e in edge_list]
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    k = len(cycles)
    
    cycle_vectors = []
    
    # Header for matrix
    print(f"\n{'='*30}\nAll Cycle Vectors Matrix\n{'='*30}")
    edge_ids = [f"e{i+1}" for i in range(m)]
    edge_strs = [str(e) for e in edge_list]
    headers_vec = ["edge_id", "edge"] + [f"C{i+1}" for i in range(k)]
    rows_vec = []
    
    # Init rows
    for i in range(m):
        rows_vec.append([edge_ids[i], edge_strs[i]])
        
    for idx, c_data in enumerate(basis_meta):
        # Reconstruct vector
        bv = c_data['bv']
        vec = np.zeros(m, dtype=int)
        for bit in range(m):
            if (bv >> bit) & 1:
                vec[bit] = 1
        cycle_vectors.append(vec)
        
        # Add to table
        for r in range(m):
            rows_vec[r].append(vec[r])
            
        # Individual print
        print(f"\nVector For Cycle {idx+1} (Length {c_data['length']}):")
        print(f"  Nodes: {sorted(list(cycles[idx]))}")
        print(f"  Edges: {[edge_ids[i] for i in range(m) if vec[i]]}")
        print(f"  Vector: {vec.tolist()}")

    # Print combined table
    print_table(headers_vec, rows_vec)
    
    # Matrices
    if k > 0:
        V = np.stack(cycle_vectors, axis=1) # m x k
        
        # Shared Edge
        S = V.T @ V
        print(f"\n{'='*30}\nShared Edge Count Matrix (C_i . C_j)\n{'='*30}")
        headers_mat = [""] + [f"C{i+1}" for i in range(k)]
        rows_S = [[f"C{i+1}"] + list(row) for i, row in enumerate(S)]
        print_table(headers_mat, rows_S)
        
        # Gram
        norms = np.linalg.norm(V, axis=0)
        norms[norms==0] = 1.0
        U = V / norms
        G_gram = U.T @ U
        
        print(f"\n{'='*30}\nGram Matrix (Cosine Similarity)\n{'='*30}")
        rows_G = []
        for i, row in enumerate(G_gram):
            fmt_row = [f"C{i+1}"] + [f"{x:.4f}" for x in row]
            rows_G.append(fmt_row)
        print_table(headers_mat, rows_G)
        
    # -----------------------------------------------

    # 3. Build Dual Graph
    DualG = build_cycle_dual_graph(G_undirected, cycles)
    print(f"\nDual Graph: {DualG.number_of_nodes()} nodes, {DualG.number_of_edges()} edges")
    
    # 4. Visualization
    if len(cycles) > 20:
         cmap = plt.cm.get_cmap('nipy_spectral', len(cycles))
    else:
         cmap = plt.cm.get_cmap('tab20', len(cycles))
         
    cycle_colors = [mcolors.to_hex(cmap(i)) for i in range(len(cycles))]
    
    # Visualize Individual Cycles
    visualize_individual_cycles(
        G_directed, G_undirected, cycles, edges_list, 
        oriented_vectors, 
        save_path="ieee30_clean_cycles.png",
        custom_colors=cycle_colors
    )
    
    # Visualize Dual Graph Comparison
    visualize_dual_comparison(
        G_directed, G_undirected, cycles, DualG,
        save_path="ieee30_clean_dual_graph.png",
        custom_colors=cycle_colors
    )
    
    # No Incidence Matrix requested.

def print_table(headers, rows):
    """Simple table printer."""
    str_rows = [[str(x) for x in row] for row in rows]
    col_widths = [len(h) for h in headers]
    for row in str_rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val))
    
    col_widths = [w + 2 for w in col_widths]
    header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    for row in str_rows:
        print("".join(f"{val:<{w}}" for val, w in zip(row, col_widths)))
    print("-" * len(header_str))

if __name__ == "__main__":
    main()
