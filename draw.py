import networkx as nx
import matplotlib.pyplot as plt

connections = [
    {"fromNeuron": "inp2", "toNeuron": "hid1", "weight": 6.2865787},
    {"fromNeuron": "inp3", "toNeuron": "hid1", "weight": 6.2865787},
    {"fromNeuron": "inp2", "toNeuron": "hid2", "weight": 6.2865787},
    {"fromNeuron": "inp3", "toNeuron": "hid3", "weight": 6.2865787},
    {"fromNeuron": "inp4", "toNeuron": "hid4", "weight": 6.2865787},
    {"fromNeuron": "inp6", "toNeuron": "hid5", "weight": 6.2865787},
    {"fromNeuron": "inpT", "toNeuron": "hidT", "weight": 6.2865787},
    {"fromNeuron": "inp2", "toNeuron": "hid6", "weight": -0.0054837866},
    {"fromNeuron": "inp3", "toNeuron": "hid6", "weight": 0.0060223085},
    {"fromNeuron": "inp4", "toNeuron": "hid6", "weight": 0.0015970375},
    {"fromNeuron": "inp5", "toNeuron": "hid6", "weight": -0.00366112},
    {"fromNeuron": "inp6", "toNeuron": "hid6", "weight": -0.002171264},
    {"fromNeuron": "inp1", "toNeuron": "hid7", "weight": 0.0021854592},
    {"fromNeuron": "inp2", "toNeuron": "hid7", "weight": 0.0001413268},
    {"fromNeuron": "inp4", "toNeuron": "hid7", "weight": -0.0025529899},
    {"fromNeuron": "inp5", "toNeuron": "hid7", "weight": 0.00033939723},
    {"fromNeuron": "inp6", "toNeuron": "hid7", "weight": -0.009290896},
    {"fromNeuron": "inp1", "toNeuron": "hid8", "weight": -0.006810256},
    {"fromNeuron": "inp2", "toNeuron": "hid8", "weight": 0.008336173},
    {"fromNeuron": "inp3", "toNeuron": "hid8", "weight": 0.0048681377},
    {"fromNeuron": "inp5", "toNeuron": "hid8", "weight": 0.007816931},
    {"fromNeuron": "inp6", "toNeuron": "hid8", "weight": -0.0087068435},
    {"fromNeuron": "inp1", "toNeuron": "hid9", "weight": -0.008006658},
    {"fromNeuron": "inp2", "toNeuron": "hid9", "weight": -0.009058826},
    {"fromNeuron": "inp3", "toNeuron": "hid9", "weight": 0.0031464088},
    {"fromNeuron": "inp4", "toNeuron": "hid9", "weight": 0.0094349105},
    {"fromNeuron": "inp6", "toNeuron": "hid9", "weight": -0.0059533417},
    {"fromNeuron": "inp1", "toNeuron": "hid10", "weight": 0.0038872538},
    {"fromNeuron": "inp3", "toNeuron": "hid10", "weight": 0.0031287645},
    {"fromNeuron": "inp4", "toNeuron": "hid10", "weight": -0.007975554},
    {"fromNeuron": "inp5", "toNeuron": "hid10", "weight": 0.009349426},
    {"fromNeuron": "inp6", "toNeuron": "hid10", "weight": 0.00035108346},
    {"fromNeuron": "inp1", "toNeuron": "hid11", "weight": 0.004403197},
    {"fromNeuron": "inp2", "toNeuron": "hid11", "weight": 0.007576909},
    {"fromNeuron": "inp3", "toNeuron": "hid11", "weight": 0.0008128891},
    {"fromNeuron": "inp4", "toNeuron": "hid11", "weight": -0.00042928848},
    {"fromNeuron": "inp5", "toNeuron": "hid11", "weight": 0.0010952014}
]

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges with weights
for connection in connections:
    from_neuron = connection["fromNeuron"]
    to_neuron = connection["toNeuron"]
    weight = connection["weight"]
    G.add_edge(from_neuron, to_neuron, weight=weight)

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

# Get weights for the edges
weights = [connection["weight"] for connection in connections]
min_weight = min(weights)
max_weight = max(weights)

# Normalize weights to the range [0.1, 5.0] for line thickness
normalized_weights = [
    0.1 + (4.9 * (weight - min_weight) / (max_weight - min_weight))
    for weight in weights
]

# Draw nodes and edges with thickness based on weights
for (from_node, to_node), weight in zip(G.edges, normalized_weights):
    nx.draw_networkx_edges(
        G, pos, edgelist=[(from_node, to_node)],
        width=weight, alpha=0.5, edge_color="gray"
    )

# Draw node labels
node_labels = {node: node for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels)

# Show the figure
plt.axis("off")
plt.show()