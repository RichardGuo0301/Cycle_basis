import osmnx as ox
import networkx as nx

point=(33.3055, -111.6815)
G = ox.graph_from_point(point, dist=800, network_type='drive_service')
try:
    G_undirected = ox.utils_graph.get_undirected(G)
except AttributeError:
    G_undirected = ox.convert.to_undirected(G)

u_target = 1649704532
v_target = 1649704546

print(f"Neighbors of {u_target}:")
if u_target in G_undirected:
    for nbr in G_undirected.neighbors(u_target):
        print(f"  - {nbr}")
else:
    print("Not found")

print(f"Neighbors of {v_target}:")
if v_target in G_undirected:
    for nbr in G_undirected.neighbors(v_target):
        print(f"  - {nbr}")
else:
    print("Not found")
