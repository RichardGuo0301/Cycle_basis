import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from cycle_basis_demo import create_random_polygon_graph
from horton_gram_mcb import find_horton_gram_mcb


def build_oriented_cycle_matrix(G_undirected, edge_list, basis_meta):
    m = len(edge_list)
    k = len(basis_meta)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}

    # Unoriented incidence from bitvector.
    C_matrix = np.zeros((m, k), dtype=int)
    for j, c_data in enumerate(basis_meta):
        for i in range(m):
            if (c_data["bv"] >> i) & 1:
                C_matrix[i, j] = 1

    # Oriented entries (+1 / -1) aligned with edge_list.
    C_oriented = np.zeros((m, k), dtype=int)
    for j, c_data in enumerate(basis_meta):
        cycle_edges = [edge_list[i] for i in range(m) if C_matrix[i, j] == 1]
        cycle_subgraph = nx.Graph(cycle_edges)
        try:
            cycle_nodes = [e[0] for e in nx.find_cycle(cycle_subgraph)]
        except Exception:
            cycle_nodes = list(cycle_subgraph.nodes())

        for i in range(len(cycle_nodes)):
            u_curr = cycle_nodes[i]
            v_curr = cycle_nodes[(i + 1) % len(cycle_nodes)]
            u, v = sorted((u_curr, v_curr))
            if (u, v) in edge_to_idx:
                idx = edge_to_idx[(u, v)]
                C_oriented[idx, j] = +1 if (u_curr == u and v_curr == v) else -1

    return C_oriented


def main():
    np.random.seed(20)
    G_directed, G_undirected = create_random_polygon_graph(12, 10, seed=100)
    pos = nx.get_node_attributes(G_undirected, "pos") or nx.spring_layout(G_undirected, seed=42)

    edge_list = sorted([tuple(sorted(e)) for e in G_undirected.edges()])
    basis_meta, _ = find_horton_gram_mcb(G_undirected, edge_list)
    C_oriented = build_oriented_cycle_matrix(G_undirected, edge_list, basis_meta)
    k = C_oriented.shape[1]

    n_panels = k + 1
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    fig.suptitle(
        f"Oriented Cycle Basis ({k} cycles)\nGreen = +1 (same dir), Red = -1 (opposite dir)",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 0: complete directed graph
    ax0 = axes[0]
    ax0.set_title("Complete Directed Graph", fontsize=10, fontweight="bold")
    nx.draw_networkx_edges(
        G_directed,
        pos,
        edge_color="gray",
        width=1.4,
        alpha=0.7,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        ax=ax0,
    )
    nx.draw_networkx_nodes(
        G_undirected,
        pos,
        node_color="#9ecae1",
        node_size=260,
        edgecolors="black",
        linewidths=1.2,
        ax=ax0,
    )
    nx.draw_networkx_labels(G_undirected, pos, font_size=8, font_weight="bold", ax=ax0)
    ax0.axis("off")

    # Panels 1..k: one oriented cycle each
    for j in range(k):
        ax = axes[j + 1]
        ax.axis("off")

        # Ghost background
        nx.draw_networkx_edges(
            G_undirected,
            pos,
            edge_color="lightgray",
            width=0.9,
            alpha=0.25,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G_undirected,
            pos,
            node_color="#e0e0e0",
            node_size=90,
            edgecolors="none",
            ax=ax,
        )

        pos_edges = []
        neg_edges = []
        active_nodes = set()

        for i, (u, v) in enumerate(edge_list):
            val = int(C_oriented[i, j])
            if val == 0:
                continue
            active_nodes.add(u)
            active_nodes.add(v)
            if val > 0:
                pos_edges.append((u, v))
            else:
                neg_edges.append((v, u))  # reverse arrow for -1

        # Draw +1 in green, -1 in red
        if pos_edges:
            nx.draw_networkx_edges(
                nx.DiGraph(pos_edges),
                pos,
                edgelist=pos_edges,
                edge_color="green",
                width=1.8,
                alpha=0.9,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=14,
                ax=ax,
            )
        if neg_edges:
            nx.draw_networkx_edges(
                nx.DiGraph(neg_edges),
                pos,
                edgelist=neg_edges,
                edge_color="red",
                width=1.8,
                alpha=0.9,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=14,
                ax=ax,
            )

        if active_nodes:
            sub = G_undirected.subgraph(active_nodes)
            nx.draw_networkx_nodes(
                sub,
                pos,
                node_color="#9ecae1",
                node_size=210,
                edgecolors="black",
                linewidths=1.2,
                ax=ax,
            )
            nx.draw_networkx_labels(sub, pos, font_size=8, font_weight="bold", ax=ax)

        green_cnt = int(np.sum(C_oriented[:, j] > 0))
        red_cnt = int(np.sum(C_oriented[:, j] < 0))
        cycle_nodes = sorted(active_nodes)
        ax.set_title(
            f"Cycle {j + 1}: {cycle_nodes}\n(Green +1: {green_cnt}, Red -1: {red_cnt})",
            fontsize=8,
            fontweight="bold",
        )

    # Hide any unused axes
    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")

    os.makedirs("visuals", exist_ok=True)
    out_path = "visuals/oriented_cycle_basis_panel.png"
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
