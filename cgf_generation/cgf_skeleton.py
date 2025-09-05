import pickle
import os
from collections import defaultdict
import torch

EXPR_DICT_PATH = 'mizar_pickles/expr_dict.pkl'
TRAIN_PATH = 'mizar_pickles/train.pkl'
OUT_PATH = 'mizar_pickles/cgf_data.pkl'

with open(EXPR_DICT_PATH, 'rb') as f:
    expr_dict = pickle.load(f)

with open(TRAIN_PATH, 'rb') as f:
    train_pairs = pickle.load(f)

co_usage = defaultdict(lambda: defaultdict(int))
for conj, premise, label in train_pairs:
    if label == 1:
        co_usage[conj][premise] += 1

cgf_dict = {}

for idx, conj in enumerate(co_usage):
    nodes = []
    edges = []

    nodes.append(expr_dict[conj])
    conj_idx = 0

    premise_nodes = {}
    idx_offset = 1

    for premise, freq in co_usage[conj].items():
        nodes.append(expr_dict[premise])
        premise_nodes[premise] = idx_offset
        edges.append((conj_idx, idx_offset))
        idx_offset += 1

    premises = list(co_usage[conj].keys())
    for i, p1 in enumerate(premises):
        for j, p2 in enumerate(premises):
            if i < j:
                edges.append((premise_nodes[p1], premise_nodes[p2]))
                edges.append((premise_nodes[p2], premise_nodes[p1]))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)

    cgf_dict[conj] = {
        'nodes': nodes,
        'edge_index': edge_index
    }

    if idx < 3:
        print(f"\nConjecture: {conj}")
        print("Sample premises:", list(co_usage[conj].keys())[:3])
        print("CGF nodes count:", len(nodes))
        print("CGF edge_index shape:", edge_index.shape)
        print("Sample edges:", edge_index[:, :min(6, edge_index.shape[1])])

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'wb') as f:
    pickle.dump(cgf_dict, f)

print(f"\nSaved CGF data for {len(cgf_dict)} conjectures to {OUT_PATH}")
