import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Imports for visualization
try:
    from cycle_dual_graph_demo import (
        build_cycle_dual_graph, 
        visualize_dual_comparison, 
        visualize_individual_cycles,
        get_cycle_edges
    )
except ImportError:
    print("Warning: Visualization modules not found. Visualization steps will be skipped.")

# ==============================================================================
# Core Algorithm: Horton + GF(2) + Gram Tie-Break
# ==============================================================================

def bit_count(x: int) -> int:
    return x.bit_count()

def cosine_bitvec(a: int, b: int) -> float:
    """Computes cosine similarity between two bitvectors (cycles)."""
    inter = bit_count(a & b)
    la = bit_count(a)
    lb = bit_count(b)
    if la == 0 or lb == 0:
        return 0.0
    return inter / math.sqrt(la * lb)

def gram_tiebreak_penalty(bv: int, selected_bitvecs):
    """
    Computes penalty for adding candidate 'bv' to existing 'selected_bitvecs'.
    Penalty tuple: (max_cosine, sum_squared_cosine)
    Lower is better (more orthogonal).
    """
    if not selected_bitvecs:
        return (0.0, 0.0)
    
    cos_vals = [cosine_bitvec(bv, s) for s in selected_bitvecs]
    max_cos = max(cos_vals) if cos_vals else 0.0
    sum_sq = sum(c*c for c in cos_vals)
    return (max_cos, sum_sq)

def get_horton_candidates(G, edge_list, edge_to_idx):
    """
    Generates ALL Horton candidates:
    For every vertex r, build BFS tree T_r.
    For every co-tree edge e=(u,v), form cycle = Path_T(r, u) + e + Path_T(v, r).
    """
    cand = {}  # bv -> (length, cycle_edges, root, chord)
    
    nodes = list(G.nodes())
    
    for r in nodes:
        # Build BFS tree rooted at r
        # Note: nx.bfs_tree returns a directed graph, we treat it as undirected structure
        T_bfs = nx.bfs_tree(G, r)
        T_edges = set()
        for u, v in T_bfs.edges():
            T_edges.add(tuple(sorted((u, v))))
            
        # Build undirected tree T for distance calculations
        T = nx.Graph()
        T.add_nodes_from(nodes)
        T.add_edges_from(T_edges)
        
        # Check all co-tree edges
        for (u, v) in edge_list:
            if (u, v) in T_edges:
                continue
                
            # Cycle = unique path in T between u and v + edge (u,v)
            try:
                path = nx.shortest_path(T, u, v)
            except nx.NetworkXNoPath:
                continue # Should not happen in connected graph
                
            # Convert path to edges
            path_edges = [tuple(sorted((a, b))) for a, b in zip(path[:-1], path[1:])]
            cycle_edges = path_edges + [(u, v)]
            
            # Create bitvector
            bv = 0
            for e in cycle_edges:
                if e in edge_to_idx:
                    bv |= (1 << edge_to_idx[e])
            
            L = bit_count(bv)
            
            # Store unique candidates. 
            # If same edge set found, keep one with "better" metadata? (doesn't matter for logic)
            if bv not in cand:
                cand[bv] = (L, cycle_edges, r, (u, v))
    
    # Return as list
    candidates = []
    for bv, (L, edges, r, chord) in cand.items():
        candidates.append({
            'bv': bv,
            'length': L,
            'edges': edges,
            'root': r,
            'chord': chord
        })
    
    # Sort primarily by length
    candidates.sort(key=lambda x: x['length'])
    return candidates

def gf2_try_add(vec: int, pivots: dict):
    """Incremental Gaussian Elimination over GF(2)"""
    x = vec
    for p in sorted(pivots.keys(), reverse=True):
        if (x >> p) & 1:
            x ^= pivots[p]
    
    if x == 0:
        return False, pivots # Dependent
    
    # Independent
    p = x.bit_length() - 1
    piv2 = dict(pivots)
    piv2[p] = x
    return True, piv2

def find_horton_gram_mcb(G, edge_list=None):
    """
    Main function to find MCB using Horton set + Gram tie-break.
    Returns: (basis_cycles, basis_bitvecs)
    """
    if edge_list is None:
        edge_list = [tuple(sorted(e)) for e in G.edges()]
    
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    n = G.number_of_nodes()
    k = m - n + nx.number_connected_components(G)
    
    print(f"Graph: n={n}, m={m}, cycle rank k={k}")
    
    # 1. Generate Candidates
    candidates = get_horton_candidates(G, edge_list, edge_to_idx)
    print(f"Generated {len(candidates)} Horton candidates.")
    
    # 2. Select Basis
    # Group by length
    groups = {}
    for c in candidates:
        groups.setdefault(c['length'], []).append(c)
        
    pivots = {}     # For GF(2) extraction
    selected_bv = [] # List of selected bitvectors
    selected_data = [] # List of selected candidate dicts
    
    for L in sorted(groups.keys()):
        if len(selected_bv) >= k:
            break
            
        pool = groups[L]
        
        while pool and len(selected_bv) < k:
            best_idx = -1
            best_key = None # (max_cos, sum_sq, bv_val)
            best_pivots = None
            
            # Iterate pool to find best candidate that is independent
            for i, cand in enumerate(pool):
                bv = cand['bv']
                
                # Check independence
                is_indep, new_pivots = gf2_try_add(bv, pivots)
                if not is_indep:
                    continue
                
                # Compute tie-break score
                # We want to minimize (max_cos, sum_sq)
                # We also append bv for deterministic breaking
                score = gram_tiebreak_penalty(bv, selected_bv) + (bv,)
                
                if best_key is None or score < best_key:
                    best_key = score
                    best_idx = i
                    best_pivots = new_pivots
            
            if best_idx == -1:
                break # No more independent cycles in this length group
                
            # Select best
            chosen = pool.pop(best_idx)
            selected_bv.append(chosen['bv'])
            selected_data.append(chosen)
            pivots = best_pivots
            
    return selected_data, edge_list

# ==============================================================================
# Helper for formatting
# ==============================================================================

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

