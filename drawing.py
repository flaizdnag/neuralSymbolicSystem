import json
import json5
import networkx as nx
import matplotlib.pyplot as plt

filename = "p20_1_0_7_06193.json"

with open(filename, "r") as json_file:
    json_load = json5.load(json_file)


inp_to_hid_connections = json_load["nn_after"]["architecture"]["inpToHidConnections"]
hid_to_out_connections = json_load["nn_after"]["architecture"]["hidToOutConnections"]
inp_layer_labels = {
    node["idx"]: node["label"]
    for node in json_load["nn_after"]["architecture"]["inpLayer"]
}
hid_layer_labels = {
    node["idx"]: node["label"]
    for node in json_load["nn_after"]["architecture"]["hidLayer"]
}
out_layer_labels = {
    node["idx"]: node["label"]
    for node in json_load["nn_after"]["architecture"]["outLayer"]
}


# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges with weights for input layer to hidden layer
for connection in inp_to_hid_connections:
    from_neuron = connection["fromNeuron"]
    to_neuron = connection["toNeuron"]
    weight = connection["weight"]
    G.add_edge(from_neuron, to_neuron, weight=weight)

# Add nodes and edges with weights for hidden layer to output layer
for connection in hid_to_out_connections:
    from_neuron = connection["fromNeuron"]
    to_neuron = connection["toNeuron"]
    weight = connection["weight"]
    G.add_edge(from_neuron, to_neuron, weight=weight)


# Set positions for the nodes
# Arrange input neurons on the left, hidden neurons in the middle, and output neurons on the right
spacing = len(hid_layer_labels) / len(inp_layer_labels)
left_pos = {
    node: (0, (i * spacing) + spacing) for i, node in enumerate(inp_layer_labels)
}
middle_pos = {node: (1, i) for i, node in enumerate(hid_layer_labels)}
right_pos = {
    node: (2, (i * spacing) + spacing) for i, node in enumerate(out_layer_labels)
}
pos = {**left_pos, **middle_pos, **right_pos}


# Get weights for the edges
weights = [
    connection["weight"]
    for connection in inp_to_hid_connections + hid_to_out_connections
]
min_weight = min(weights)
max_weight = max(weights)

# Normalize weights for line thinckness from 1 to 5
prenormalized_weights = [weight + abs(min_weight) for weight in weights]
print(prenormalized_weights)
normalized_weights = [
    ((weight * 4) / max(prenormalized_weights)) + 1 for weight in prenormalized_weights
]
print(normalized_weights)


# Draw nodes and edges with thickness based on weights
plt.figure(figsize=(30, 10))
nx.draw_networkx_nodes(G, pos, node_size=200, node_color="lightblue")

for edge in G.edges:
    from_node, to_node = edge
    weight = G[from_node][to_node]["weight"]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(from_node, to_node)],
        width=weight + 0.1,
        alpha=0.5,
        edge_color="gray",
    )


# Draw node labels
node_labels = {**inp_layer_labels, **hid_layer_labels, **out_layer_labels}
node_labels = {node: label for node, label in node_labels.items() if node in G.nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="black")

# Draw edge labels (weights)
edge_labels = {
    (edge[0], edge[1]): f'{G[edge[0]][edge[1]]["weight"]:.4f}' for edge in G.edges
}
edge_label_pos = {k: (v[0], v[1]) for k, v in pos.items()}
# edge_label_pos = {(edge[0], edge[1]): ((pos[edge[0]][0] + pos[edge[1]][0]) / 2, (pos[edge[0]][1] + pos[edge[1]][1]) / 2) for edge in G.edges}
nx.draw_networkx_edge_labels(G, edge_label_pos, edge_labels=edge_labels, font_size=4)


plt.axis("off")
plt.title("Neural Network Visualization")
# plt.show()
plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
plt.close()
