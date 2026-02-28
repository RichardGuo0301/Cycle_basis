"""
Find Alternative Minimum Cycles
Analyzes the Cycle Space to find 'competitor' cycles that are as short as the chosen basis cycles
but were not selected.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Use the Fig 1 graph generator
from fig1_paper_demo import create_fig1_graph

def find_alternative_cycles():
    print("="*60)
    print("Analyzing Alternative Minimum Cycles for Fig 1")
    print("="*60)
    
    # 1. Get the Graph
    _, G = create_fig1_graph()
    
    # 2. Get Standard Minimum Cycle Basis
    basis = nx.minimum_cycle_basis(G)
    print(f"\nStandard Basis ({len(basis)} cycles):")
    basis_lengths = [len(c) for c in basis]
    for i, c in enumerate(basis):
        print(f"  Basis {i+1}: {sorted(c)} (Length {len(c)})")
        
    print(f"\nBasis Length Distribution: {sorted(basis_lengths)}")
    
    # 3. Find ALL Simple Cycles (up to a reasonable length, e.g., max basis length + 2)
    max_len = max(basis_lengths) + 2
    print(f"\nSearching for all simple cycles up to length {max_len}...")
    
    # nx.simple_cycles is for directed. For undirected, use cycle_basis helper or basic recursion?
    # nx.simple_cycles(G) works on undirected graphs in newer versions by converting to directed? 
    # No, strictly simple_cycles is for DiGraph. For Graph, we can cycle_basis but that's what we have.
    # Let's use a custom DFS or find_cycle?
    # Actually, for small graphs, we can find all cycles.
    
    all_cycles = []
    # Using a known algorithm for all simple cycles in undirected graph is distinct.
    # Let's simple_cycles on G.to_directed() and filter for uniqueness (u-v-w and w-v-u are same)
    
    G_dir = G.to_directed()
    raw_cycles = list(nx.simple_cycles(G_dir))
    
    # Filter unique undirected cycles
    unique_cycles = set()
    cleaned_cycles = []
    
    for c in raw_cycles:
        if len(c) < 3: continue
        # Normalize: Rotate to start with min node, check forward/backward
        c_min_idx = c.index(min(c))
        c_rot = c[c_min_idx:] + c[:c_min_idx]
        
        # Check reverse too
        c_rev = c_rot[::-1]
        c_rev = c_rev[-1:] + c_rev[:-1] # adjust rotation for reverse? No, just canonicalize
        
        # Canonical tuple: min((rot), (rev_rot)) is hard.
        # Easier: frozenset of edges? No, 8-figure cycles exist.
        # For simple cycles, set of edges is unique representation.
        
        # Canonical string for set check
        # Rotate starting at min, go in direction of smaller neighbor
        n = len(c)
        min_node = min(c)
        idx = c.index(min_node)
        
        # Forward
        fw = c[idx:] + c[:idx]
        # Backward
        bw = fw[0:1] + fw[1:][::-1]
        
        if fw[1] < bw[1]:
            canonical = tuple(fw)
        else:
            canonical = tuple(bw)
            
        if canonical not in unique_cycles:
            unique_cycles.add(canonical)
            cleaned_cycles.append(list(canonical))
            
    print(f"Found {len(cleaned_cycles)} unique simple cycles in total.")
    
    # 4. Compare against Basis lengths
    # We care about cycles that distinct from Basis but have same lengths as some basis cycles.
    
    # Convert Basis to canonical tuples for set subtraction
    basis_set = set()
    for c in basis:
        n = len(c)
        min_node = min(c)
        idx = c.index(min_node)
        fw = c[idx:] + c[:idx]
        bw = fw[0:1] + fw[1:][::-1]
        if fw[1] < bw[1]: canonical = tuple(fw)
        else: canonical = tuple(bw)
        basis_set.add(canonical)
        
    alternatives = []
    for c in cleaned_cycles:
        c_tuple = tuple(c) # Assuming cleanup did canonicalization
        
        # Re-canonicalize just in case to match basis_set logic 100%
        n = len(c)
        min_node = min(c)
        idx = c.index(min_node)
        fw = c[idx:] + c[:idx]
        bw = fw[0:1] + fw[1:][::-1]
        if fw[1] < bw[1]: canonical = tuple(fw)
        else: canonical = tuple(bw)
        
        if canonical in basis_set:
            continue # It IS in the basis
            
        length = len(c)
        # Is this length a "Basis Length"? (i.e., could it compete?)
        # Or even strictly: Is it equal to the length of *some* basis element?
        # A cycle of length 5 is interesting if we have Basis elements of length 5.
        
        if length in basis_lengths:
            print(f"  candidate: {sorted(c)} (Length {length})")
            alternatives.append(c)
            
    print(f"\nFound {len(alternatives)} alternative cycles with lengths matching basis cycles.")
    
    # Detailed Check for Length 5 (The Case of C3)
    target_len = 5
    print(f"\n--- Deep Dive: Cycles of Length {target_len} ---")
    basis_len5 = [c for c in basis if len(c) == target_len]
    alts_len5 = [c for c in alternatives if len(c) == target_len]
    
    print(f"Basis has {len(basis_len5)} cycles of length {target_len}:")
    for c in basis_len5:
        print(f"  Basis: {c}")
        
    print(f"Alternatives found ({len(alts_len5)}):")
    for c in alts_len5:
        print(f"  Alt  : {c}")
        
if __name__ == "__main__":
    find_alternative_cycles()
