import json
import json5
import networkx as nx
import matplotlib.pyplot as plt


with open("p20_0_1_1_00037.json", "r") as json_file:
	json_load = json5.load(json_file)

# print(json_load["nn_after"]["architecture"])
nn_after = json_load["nn_after"]["architecture"]
connections = json_load["nn_after"]["architecture"]["inpToHidConnections"]
labels = json_load["nn_after"]["architecture"]["inpLayer"]

# Create a directed graph
G = nx.DiGraph()

for connection in connections:
    from_neuron = connection["fromNeuron"]
    to_neuron = connection["toNeuron"]
    weight = connection["weight"]
    G.add_edge(from_neuron, to_neuron, weight=weight)

# min and max weights
weights = [connection["weight"] for connection in connections]
min_weight = min(weights)
max_weight = max(weights)

# Normalize weights for line thickness
normalized_weights = [
    0.1 + (4.9 * (weight - min_weight) / (max_weight - min_weight))
    for weight in weights
]
# Set positions for the nodes
pos = {}
x = 1  # x-coordinate for "inp" nodes
y = 1  # y-coordinate for "hid" nodes
for node in G.nodes:
    if node.startswith("inp"):
        pos[node] = (x, y)
        y -= 1
    elif node.startswith("hid"):
        pos[node] = (x + 2, y)
        y -= 1
        
# Draw nodes and edges with thickness based on weights
for (from_node, to_node), weight in zip(G.edges, normalized_weights):
    nx.draw_networkx_edges(
        G, pos, edgelist=[(from_node, to_node)],
        width=weight, alpha=0.5, edge_color="gray"
    )

# Draw node labels
node_labels = {node: "" for node in G.nodes}
for item in labels:
    node_idx = item["idx"]
    node_label = item["label"]
    node_labels[node_idx] = node_label
nx.draw_networkx_labels(G, pos, labels=node_labels)

# Show the figure
plt.axis("off")
plt.show()