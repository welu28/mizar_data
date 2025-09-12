import pickle
import os
import torch
from collections import defaultdict
from itertools import combinations

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_embeddings(expr_dict, vocab):
    embeddings = {}
    for fid, expr in expr_dict.items():
        embeddings[fid] = torch.randn(128)
    return embeddings

class ASTNode:
    def __init__(self, label, node_type="ast"):
        self.label = label
        self.type = node_type
        self.children = []

def walk_ast(expr, parent_id, nodes, edges, next_id):
    current_id = next_id
    node = ASTNode(str(expr), "ast")
    nodes[current_id] = node

    if parent_id is not None:
        edges.append((parent_id, current_id))

    if isinstance(expr, (list, tuple)):
        for child in expr:
            next_id = walk_ast(child, current_id, nodes, edges, next_id + 1)

    return next_id

def build_ast_edges(fid, expr):
    nodes = {}
    edges = []
    walk_ast(expr, None, nodes, edges, 0)
    return nodes, edges

def compute_co_usage(pairs):
    co_usage = defaultdict(int)
    conj_to_premises = defaultdict(list)

    for conj, premise, label in pairs:
        if label == 1:
            conj_to_premises[conj].append(premise)

    for conj, premises in conj_to_premises.items():
        for (p1, p2) in combinations(premises, 2):
            co_usage[tuple(sorted((p1, p2)))] += 1

    return co_usage, conj_to_premises

def build_cgf(conjecture, premises, embeddings, expr_dict, co_usage):
    nodes = {}
    edges = []

    nodes[conjecture] = {
        "embedding": embeddings[conjecture],
        "type": "conjecture"
    }

    for p in premises:
        nodes[p] = {
            "embedding": embeddings[p],
            "type": "premise"
        }

        ast_nodes, ast_edges = build_ast_edges(p, expr_dict[p])
        for nid, ast_node in ast_nodes.items():
            nodes[(p, nid)] = {
                "label": ast_node.label,
                "type": ast_node.type
            }
        for (u, v) in ast_edges:
            edges.append({
                "src": (p, u),
                "dst": (p, v),
                "type": "ast",
                "weight": 1.0
            })

        edges.append({"src": conjecture, "dst": p,
                      "type": "conjecture-premise", "weight": 1.0})

    for (p1, p2) in combinations(premises, 2):
        if (p1, p2) in co_usage:
            weight = float(co_usage[(p1, p2)])
            edges.append({"src": p1, "dst": p2,
                          "type": "co_usage", "weight": weight})
            edges.append({"src": p2, "dst": p1,
                          "type": "co_usage", "weight": weight})

    return {"nodes": nodes, "edges": edges}

def main(data_dir="mizar_pickles/"):
    out_dir = os.path.join(data_dir, "cgf_pickles")
    os.makedirs(out_dir, exist_ok=True)

    train = load_pickle(os.path.join(data_dir, "train.pkl"))
    val = load_pickle(os.path.join(data_dir, "val.pkl"))
    test = load_pickle(os.path.join(data_dir, "test.pkl"))
    expr_dict = load_pickle(os.path.join(data_dir, "expr_dict.pkl"))
    vocab = load_pickle(os.path.join(data_dir, "vocab.pkl"))

    embeddings = load_embeddings(expr_dict, vocab)
    co_usage, conj_to_premises = compute_co_usage(train)

    datasets = {"train": train, "val": val, "test": test}
    for split, pairs in datasets.items():
        cgf_graphs = []
        seen_conj = set()

        for conj, premise, label in pairs:
            if conj not in seen_conj:
                premises = conj_to_premises.get(conj, [])
                cgf_graphs.append(build_cgf(conj, premises, embeddings, expr_dict, co_usage))
                seen_conj.add(conj)

        with open(os.path.join(out_dir, f"cgf_{split}.pkl"), "wb") as f:
            pickle.dump(cgf_graphs, f)
        print(f"Built {len(cgf_graphs)} CGF graphs ({split} set).")

if __name__ == "__main__":
    main()