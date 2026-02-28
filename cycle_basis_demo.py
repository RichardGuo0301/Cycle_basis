"""
Cycle Basis Utility Functions for Robot Path Planning
Core functions for creating directed graphs and computing cycle basis.
Visualization is handled by cycle_dual_graph_demo.py
"""

import networkx as nx
import random
import math
import numpy as np


def create_directed_grid_graph(rows: int, cols: int) -> tuple:
    """
    Create a directed grid graph with single-direction edges.
    Returns both the DiGraph and its underlying undirected Graph.
    
    Edge orientation pattern (forms clockwise cycles in each 2x2 cell):
    - Even rows: right →
    - Odd rows: left ←
    - Even cols: down ↓
    - Odd cols: up ↑
    
    Returns:
        (G_directed, G_undirected): Tuple of directed and undirected graphs
    """
    G_directed = nx.DiGraph()
    
    # Add nodes with positions
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            G_directed.add_node(node_id, pos=(c, rows - 1 - r))
    
    # Add single-direction edges
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            
            # Horizontal edges
            if c < cols - 1:
                if r % 2 == 0:
                    G_directed.add_edge(node, node + 1)  # Even rows: right →
                else:
                    G_directed.add_edge(node + 1, node)  # Odd rows: left ←
            
            # Vertical edges
            if r < rows - 1:
                if c % 2 == 0:
                    G_directed.add_edge(node, node + cols)  # Even cols: down ↓
                else:
                    G_directed.add_edge(node + cols, node)  # Odd cols: up ↑
    
    # Create underlying undirected graph
    G_undirected = G_directed.to_undirected()
    
    return G_directed, G_undirected


def get_oriented_cycle_vectors(G_directed: nx.DiGraph, cycles: list) -> list:
    """
    Compute oriented cycle vectors for each cycle.
    For each edge in cycle:
      c_i = +1 if edge aligns with cycle direction
      c_i = -1 if edge opposes cycle direction
      c_i = 0 if edge not in cycle
    
    Returns:
        List of dicts: {edge: orientation} for each cycle
    """
    oriented_vectors = []
    
    for cycle in cycles:
        vector = {}
        n = len(cycle)
        
        for i in range(n):
            u, v = cycle[i], cycle[(i + 1) % n]
            
            # Check edge direction in the directed graph
            if G_directed.has_edge(u, v):
                vector[(u, v)] = +1  # Edge aligns with cycle direction
            elif G_directed.has_edge(v, u):
                vector[(v, u)] = -1  # Edge opposes cycle direction
        
        oriented_vectors.append(vector)
    
    return oriented_vectors


def get_cycle_basis_info(G_undirected: nx.Graph, G_directed: nx.DiGraph = None):
    """
    Get minimum cycle basis from UNDIRECTED graph.
    If directed graph provided, also compute oriented cycle vectors.
    
    Args:
        G_undirected: Undirected graph for cycle basis computation
        G_directed: Optional directed graph for oriented vector computation
    
    Returns:
        (cycles, cycle_nodes, cycle_edges, oriented_vectors)
    """
    cycles = nx.minimum_cycle_basis(G_undirected)
    
    cycle_nodes = []
    cycle_edges = []
    
    for cycle in cycles:
        nodes = set(cycle)
        cycle_nodes.append(nodes)
        
        # Find actual edges between cycle nodes
        edges = set()
        for u in cycle:
            for v in cycle:
                if u < v and G_undirected.has_edge(u, v):
                    edges.add((u, v))
        cycle_edges.append(edges)
    
    # Compute oriented vectors if directed graph provided
    oriented_vectors = None
    if G_directed is not None:
        oriented_vectors = get_oriented_cycle_vectors(G_directed, cycles)
    
    return cycles, cycle_nodes, cycle_edges, oriented_vectors


def create_random_polygon_graph(n_nodes: int, n_internal_edges: int, seed: int = 42) -> tuple:
    """
    Create a directed polygon graph with internal diagonals.
    Returns both directed and undirected versions.
    
    Args:
        n_nodes: Number of nodes arranged in a circle
        n_internal_edges: Number of random internal diagonal edges
        seed: Random seed for reproducibility
    
    Returns:
        (G_directed, G_undirected): Tuple of directed and undirected graphs
    """
    random.seed(seed)
    G_undirected = nx.Graph()
    
    # 1. Generate nodes on a circle
    pos = {}
    radius = 10
    angle_step = 2 * math.pi / n_nodes
    
    for i in range(n_nodes):
        angle = i * angle_step
        pos[i] = (radius * math.cos(angle), radius * math.sin(angle))
        G_undirected.add_node(i)
        
    nx.set_node_attributes(G_undirected, pos, 'pos')
    
    # 2. Create polygon boundary (Hamiltonian cycle)
    for i in range(n_nodes):
        G_undirected.add_edge(i, (i + 1) % n_nodes)
        
    # 3. Add random internal diagonals
    possible_diagonals = []
    for i in range(n_nodes):
        for j in range(i + 2, n_nodes):
            if i == 0 and j == n_nodes - 1:
                continue
            possible_diagonals.append((i, j))
            
    if possible_diagonals:
        n_add = min(n_internal_edges, len(possible_diagonals))
        chosen = random.sample(possible_diagonals, n_add)
        G_undirected.add_edges_from(chosen)
    
    # 4. Create directed version with alternating directions
    G_directed = nx.DiGraph()
    for node, data in G_undirected.nodes(data=True):
        G_directed.add_node(node, **data)
    
    for u, v in G_undirected.edges():
        # Assign direction based on node parity
        if (u + v) % 2 == 0:
            G_directed.add_edge(u, v)
        else:
            G_directed.add_edge(v, u)
    
    return G_directed, G_undirected


def get_oriented_incidence_matrix(G_directed: nx.DiGraph):
    """
    Compute the oriented incidence matrix I^o (n x m).
    
    Definition from paper:
    I^o_ij = 1 if arc e_j leaves node v_i
    I^o_ij = -1 if arc e_j enters node v_i
    I^o_ij = 0 otherwise
    
    Returns:
        I_o (numpy.ndarray): The incidence matrix
        node_order (list): List of nodes corresponding to rows
        edge_order (list): List of edges corresponding to columns
    """
    nodes = sorted(list(G_directed.nodes()))
    edges = sorted(list(G_directed.edges()))
    
    n = len(nodes)
    m = len(edges)
    
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    I_o = np.zeros((n, m), dtype=int)
    
    for j, (u, v) in enumerate(edges):
        # Edge e_j = (u, v) goes from u to v
        # Leaves u => I^o_uj = 1
        # Enters v => I^o_vj = -1
        
        row_u = node_to_idx[u]
        row_v = node_to_idx[v]
        
        I_o[row_u, j] = 1
        I_o[row_v, j] = -1
        
    return I_o, nodes, edges
