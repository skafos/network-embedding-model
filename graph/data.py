import json
from ast import literal_eval as make_tuple


def load_nodes():
    with open("graph/nodes.txt", 'r') as f:
        node_weights = json.load(f)
    return dict([(int(k), float(v)) for k, v in node_weights.items()])

def load_edges():
    with open("graph/edges.txt", 'r') as f:
        edge_weights = json.load(f)
    return dict([(make_tuple(k), float(v)) for k, v in edge_weights.items()])
