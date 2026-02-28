import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.lines import Line2D

def make_gif(path_edges, euler_path_original, num_cycles, output_file, 
             G_undirected, pos, edge_tau_dict, nodes, node_to_idx, latency_requirements, title_prefix="", path_colors=None, fps=2):
    """
    Generate patrol animation GIF for an Eulerian circuit.
    """
    fig_a, ax_a = plt.subplots(figsize=(14, 10))
    n_edges = len(path_edges)
    single_len = len(euler_path_original)

    def update(frame):
        ax_a.clear()
        edge_visit_counts = {}
        cumulative_time = 0.0
        node_last_visit = {v: 0.0 for v in nodes}

        current_edge_idx = min(frame, n_edges - 1)
        for idx in range(current_edge_idx + 1):
            u, v = path_edges[idx]
            key = tuple(sorted((u, v)))
            edge_visit_counts[key] = edge_visit_counts.get(key, 0) + 1
            if idx < current_edge_idx:
                cumulative_time += edge_tau_dict[(u, v)]
                node_last_visit[v] = cumulative_time

        current_time = cumulative_time

        node_urgency = {}
        for node in nodes:
            time_since = current_time - node_last_visit[node]
            node_latency = latency_requirements[node_to_idx[node]]
            urgency = min(time_since / node_latency, 1.0)
            node_urgency[node] = urgency

        node_colors = [plt.cm.RdYlGn_r(node_urgency[v]) for v in nodes]

        nx.draw_networkx_edges(G_undirected, pos, width=1, alpha=0.15, edge_color="gray", ax=ax_a)
        if frame > 0:
            for (u_p, v_p), count in edge_visit_counts.items():
                width = 2 + count * 0.8
                nx.draw_networkx_edges(G_undirected, pos, edgelist=[(u_p, v_p)],
                                       width=width, edge_color="blue", alpha=0.6, ax=ax_a)

        if frame < n_edges:
            if n_edges > 0:
                u, v = path_edges[current_edge_idx]
                edge_c = "red"
                if path_colors and current_edge_idx < len(path_colors):
                    edge_c = path_colors[current_edge_idx]
                nx.draw_networkx_edges(G_undirected, pos, [(u, v)], width=5,
                                       edge_color=edge_c, ax=ax_a, alpha=0.9)

        nx.draw_networkx_nodes(G_undirected, pos, node_color=node_colors,
                               node_size=100, ax=ax_a, edgecolors="black", linewidths=1.0)

        labels = {}
        for v in nodes:
            time_since = current_time - node_last_visit[v]
            node_latency = latency_requirements[node_to_idx[v]]
            # Only draw time/latency text, not the node ID, 
            # and maybe keep it small
            # labels[v] = f"{time_since:.1f}/{node_latency:.0f}"
            pass # Or just completely remove labels if it gets too messy. Let's just remove them for clarity.
        # nx.draw_networkx_labels(G_undirected, pos, labels, font_size=6, font_weight="bold", ax=ax_a)

        edge_labels_dict = {
            (u, v): f"{edge_tau_dict.get((u, v), edge_tau_dict.get((v, u), 0)):.1f}s"
            for u, v in G_undirected.edges()
        }
        nx.draw_networkx_edge_labels(G_undirected, pos, edge_labels_dict, font_size=8, ax=ax_a)

        if frame < n_edges:
            u, _ = path_edges[current_edge_idx]
            ax_a.plot(pos[u][0], pos[u][1], "ro", markersize=20, zorder=10,
                      markeredgecolor="white", markeredgewidth=3)
        elif n_edges > 0:
            last_node = path_edges[-1][1]
            ax_a.plot(pos[last_node][0], pos[last_node][1], "ro", markersize=20,
                      zorder=10, markeredgecolor="white", markeredgewidth=3)

        cycle_num = frame // single_len + 1 if single_len > 0 else 1
        step_in_cycle = frame % single_len if single_len > 0 else 0
        T_tour_single = sum(edge_tau_dict[e] for e in euler_path_original)
        
        tag_info = ""
        if path_colors and current_edge_idx < len(path_colors):
            c = path_colors[current_edge_idx]
            if c == "limegreen":
                tag_info = " [+ DETOUR]"
            elif c == "royalblue":
                tag_info = " [MATCH]"
                
        ax_a.set_title(
            f"{title_prefix}Latency Animation{tag_info}\n"
            f"T_tour={T_tour_single:.1f}s\n"
            f"Time: {current_time:.1f}s | "
            f"Cycle {cycle_num}/{num_cycles} Step {step_in_cycle}/{single_len}",
            fontsize=14, fontweight="bold",
        )
        ax_a.axis("off")

    frames = n_edges + 10
    interval_ms = int(1000 / fps) if fps > 0 else 500
    anim = animation.FuncAnimation(fig_a, update, frames=frames, interval=interval_ms, repeat=True)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer="pillow", fps=fps)
    print(f"Saved: {output_file}")
    plt.close(fig_a)


def plot_static_correction(G_undirected, pos, edge_list, flow_B, Cm, blocked_node, blocked_edge, incident_indices, blocked_idx, balanced, is_weakly_conn, nodes, output_file):
    """
    Generate static correction diagram showing original flow, blocked items, and Cm correction.
    - If blocked_node is not None, it's a node blocking plot.
    - Else it's an edge blocking plot.
    """
    fig_s, ax_s = plt.subplots(figsize=(14, 10))

    # Draw all edges in light gray as base
    nx.draw_networkx_edges(G_undirected, pos, width=1, alpha=0.15, edge_color="gray", ax=ax_s)

    # Draw original active edges (flow_B != 0) in blue with arrows
    DG_orig = nx.DiGraph()
    for i, (u, v) in enumerate(edge_list):
        f = int(flow_B[i])
        if f > 0:
            DG_orig.add_edge(u, v, weight=abs(f))
        elif f < 0:
            DG_orig.add_edge(v, u, weight=abs(f))

    orig_edges = list(DG_orig.edges())
    orig_widths = [1.5 + DG_orig[u][v]["weight"] * 0.5 for u, v in orig_edges]
    nx.draw_networkx_edges(DG_orig, pos, edgelist=orig_edges, width=orig_widths,
                           edge_color="royalblue", alpha=0.4, arrows=True,
                           arrowstyle="-|>", arrowsize=15, ax=ax_s,
                           connectionstyle="arc3,rad=0.1")

    legend_elements = [
        Line2D([0], [0], color="royalblue", linewidth=2, alpha=0.5, label="Original flow_B"),
    ]

    # Draw blocked elements
    if blocked_node is not None:
        for i in incident_indices:
            bu, bv = edge_list[i]
            nx.draw_networkx_edges(G_undirected, pos, edgelist=[(bu, bv)], width=6,
                                   edge_color="black", alpha=0.9, ax=ax_s, style="dashed")
        legend_elements.append(Line2D([0], [0], color="black", linewidth=4, linestyle="--", label=f"Blocked node {blocked_node} edges ({len(incident_indices)})"))
        excluded_cm_indices = set(incident_indices)
    else:
        # Support single edge (tuple) or multiple edges (list of tuples)
        blocked_edges_list = blocked_edge if isinstance(blocked_edge, list) else [blocked_edge]
        blocked_idx_set    = set(blocked_idx) if isinstance(blocked_idx, list) else {blocked_idx}
        for be in blocked_edges_list:
            bu, bv = be
            nx.draw_networkx_edges(G_undirected, pos, edgelist=[(bu, bv)], width=6,
                                   edge_color="black", alpha=0.9, ax=ax_s, style="dashed")
        label_str = ", ".join(str(be) for be in blocked_edges_list)
        legend_elements.append(Line2D([0], [0], color="black", linewidth=4, linestyle="--",
                                      label=f"Blocked edge(s): {label_str}"))
        excluded_cm_indices = blocked_idx_set

    # Draw Cm correction edges
    for i, (u, v) in enumerate(edge_list):
        cm_val = int(Cm[i])
        if cm_val == 0 or i in excluded_cm_indices:
            continue
        if cm_val > 0:
            nx.draw_networkx_edges(nx.DiGraph([(u, v)]), pos, edgelist=[(u, v)],
                                   width=3 + abs(cm_val), edge_color="green", alpha=0.8,
                                   arrows=True, arrowstyle="-|>", arrowsize=20, ax=ax_s,
                                   connectionstyle="arc3,rad=0.15")
        else:
            nx.draw_networkx_edges(nx.DiGraph([(v, u)]), pos, edgelist=[(v, u)],
                                   width=3 + abs(cm_val), edge_color="orange", alpha=0.8,
                                   arrows=True, arrowstyle="-|>", arrowsize=20, ax=ax_s,
                                   connectionstyle="arc3,rad=0.15")

    legend_elements.append(Line2D([0], [0], color="green", linewidth=3, label="Cm: added flow (+)"))
    legend_elements.append(Line2D([0], [0], color="orange", linewidth=3, label="Cm: added flow (-)"))

    # Draw nodes
    if blocked_node is not None:
        node_colors_s = ["tomato" if v == blocked_node else "lightblue" for v in nodes]
        legend_elements.append(Line2D([0], [0], marker="o", color="tomato", markersize=12, linestyle="None", label=f"Blocked node {blocked_node}"))
    else:
        node_colors_s = "lightblue"
        
    nx.draw_networkx_nodes(G_undirected, pos, node_color=node_colors_s,
                           node_size=100, ax=ax_s, edgecolors="black", linewidths=1.0)
    # Disabled node labels
    # nx.draw_networkx_labels(G_undirected, pos, font_size=11, font_weight="bold", ax=ax_s)

    # Edge labels
    edge_annot = {}
    for i, (u, v) in enumerate(edge_list):
        parts = []
        if flow_B[i] != 0:
            parts.append(f"f={flow_B[i]}")
        if Cm[i] != 0:
            parts.append(f"Cm={Cm[i]:+d}")
        if parts:
            edge_annot[(u, v)] = "\n".join(parts)
    nx.draw_networkx_edge_labels(G_undirected, pos, edge_annot, font_size=7, ax=ax_s)

    ax_s.legend(handles=legend_elements, loc="upper left", fontsize=11)
    
    if blocked_node is not None:
        title = (f"Node Blocking Correction: block node {blocked_node} ({len(incident_indices)} incident edges)\n"
                 f"||Cm||_1={int(np.sum(np.abs(Cm)))}  |  Eulerian: {balanced and is_weakly_conn}")
    else:
        blocked_edges_list = blocked_edge if isinstance(blocked_edge, list) else [blocked_edge]
        edges_str = ", ".join(str(be) for be in blocked_edges_list)
        title = (f"Edge Blocking Correction: block edge(s) {edges_str}\n"
                 f"||Cm||_1={int(np.sum(np.abs(Cm)))}  |  Eulerian: {balanced and is_weakly_conn}")
                 
    ax_s.set_title(title, fontsize=14, fontweight="bold")
    ax_s.axis("off")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    print(f"Saving static diagram to {output_file}...")
    fig_s.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig_s)


def plot_corrected_result(G_undirected, pos, edge_list, flow_corrected, Cm, blocked_node, blocked_edge, incident_indices, blocked_idx, balanced, is_weakly_conn, nodes, output_file):
    """
    Generate clean two-color corrected flow result diagram.
    """
    fig_r, ax_r = plt.subplots(figsize=(14, 10))

    nx.draw_networkx_edges(G_undirected, pos, width=1, alpha=0.15, edge_color="gray", ax=ax_r)

    DG_result = nx.DiGraph()
    for i, (u, v) in enumerate(edge_list):
        fc = int(flow_corrected[i])
        if fc > 0:
            DG_result.add_edge(u, v, weight=abs(fc))
        elif fc < 0:
            DG_result.add_edge(v, u, weight=abs(fc))

    result_edges = list(DG_result.edges())
    result_widths = [1.5 + DG_result[u][v]["weight"] * 0.9 for u, v in result_edges]
    nx.draw_networkx_edges(DG_result, pos, edgelist=result_edges, width=result_widths,
                           edge_color="royalblue", alpha=0.85, arrows=True,
                           arrowstyle="-|>", arrowsize=18, ax=ax_r,
                           connectionstyle="arc3,rad=0.1")

    legend_r = [Line2D([0], [0], color="royalblue", linewidth=3, label="Corrected flow (flow_corrected ≠ 0)")]

    if blocked_node is not None:
        for i in incident_indices:
            buu, bvv = edge_list[i]
            nx.draw_networkx_edges(G_undirected, pos, edgelist=[(buu, bvv)], width=5,
                                   edge_color="black", alpha=0.9, ax=ax_r, style="dashed")
        legend_r.append(Line2D([0], [0], color="black", linewidth=4, linestyle="--", label=f"Blocked node {blocked_node} edges"))
        node_colors_r = ["tomato" if v == blocked_node else "lightblue" for v in nodes]
    else:
        # Support single edge (tuple) or multiple edges (list of tuples)
        blocked_edges_list = blocked_edge if isinstance(blocked_edge, list) else [blocked_edge]
        for be in blocked_edges_list:
            buu, bvv = be
            nx.draw_networkx_edges(G_undirected, pos, edgelist=[(buu, bvv)], width=5,
                                   edge_color="black", alpha=0.9, ax=ax_r, style="dashed")
        label_str = ", ".join(str(be) for be in blocked_edges_list)
        legend_r.append(Line2D([0], [0], color="black", linewidth=4, linestyle="--",
                                label=f"Blocked edge(s): {label_str}"))
        node_colors_r = "lightblue"

    nx.draw_networkx_nodes(G_undirected, pos, node_color=node_colors_r,
                           node_size=100, ax=ax_r, edgecolors="black", linewidths=1.0)
    # Disabled node labels
    # nx.draw_networkx_labels(G_undirected, pos, font_size=11, font_weight="bold", ax=ax_r)

    result_annot = {}
    for i, (u, v) in enumerate(edge_list):
        fc = int(flow_corrected[i])
        if fc != 0:
            result_annot[(u, v)] = f"f={fc}"
    nx.draw_networkx_edge_labels(G_undirected, pos, result_annot, font_size=8, ax=ax_r)

    ax_r.legend(handles=legend_r, loc="upper left", fontsize=11)
    
    if blocked_node is not None:
        title = (f"Corrected Flow After Blocking Node {blocked_node}\n"
                 f"||Cm||_1={int(np.sum(np.abs(Cm)))}  |  Eulerian: {balanced and is_weakly_conn}  |  Active edges: {len(result_edges)}")
    else:
        blocked_edges_list = blocked_edge if isinstance(blocked_edge, list) else [blocked_edge]
        edges_str = ", ".join(str(be) for be in blocked_edges_list)
        title = (f"Corrected Flow After Blocking Edge(s) {edges_str}\n"
                 f"||Cm||_1={int(np.sum(np.abs(Cm)))}  |  Eulerian: {balanced and is_weakly_conn}  |  Active edges: {len(result_edges)}")
                 
    ax_r.set_title(title, fontsize=14, fontweight="bold")
    ax_r.axis("off")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    print(f"Saving corrected result diagram to {output_file}...")
    fig_r.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig_r)
