import pickle
import os
import torch
from collections import defaultdict
from itertools import combinations

# --- CGF utilities for dict-based ASTs ---
def flatten_ast(ast_dict):
    """Recursively flatten a dict-based AST into a list of nodes."""
    nodes = [ast_dict]
    for child in ast_dict.get("children", []):
        nodes.extend(flatten_ast(child))
    return nodes

def build_embeddings_from_ast(expr_dict):
    """Create embeddings for all AST nodes in expr_dict."""
    embeddings = {}
    for fid, ast_root in expr_dict.items():
        ast_nodes = flatten_ast(ast_root)       # flatten dict AST to list
        embeddings[fid] = torch.randn(len(ast_nodes), 128)  # one embedding per node
    return embeddings

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

def build_cgf(conj, premises, embeddings, expr_dict, co_usage):
    nodes = {}
    edges = {}

    # Conjecture node
    nodes[conj] = {"type": "conjecture"}

    for p in premises:
        nodes[p] = {"type": "premise"}

        ast_root = expr_dict[p]             # already dict-based AST
        ast_nodes = flatten_ast(ast_root)

        for i, n in enumerate(ast_nodes):
            nodes[(p, i)] = {
                "label": n.get("value", str(n)),
                "type": "ast",
                "embedding": embeddings[p][i]
            }

        # build AST edges
        for i, n in enumerate(ast_nodes):
            for child in n.get("children", []):
                j = ast_nodes.index(child)
                edges[(p, i, p, j)] = {"type": "ast", "weight": 1.0}

        # conjecture -> premise
        edges[(conj, p)] = {"type": "conjecture-premise", "weight": 1.0}

    # co-usage edges
    for (p1, p2) in combinations(premises, 2):
        if (p1, p2) in co_usage:
            weight = float(co_usage[(p1, p2)])
            edges[(p1, p2)] = {"type": "co_usage", "weight": weight}
            edges[(p2, p1)] = {"type": "co_usage", "weight": weight}

    return {"nodes": nodes, "edges": edges}

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main(data_dir="mizar_pickles/"):
    out_dir = os.path.join(data_dir, "cgf_pickles")
    os.makedirs(out_dir, exist_ok=True)

    train = load_pickle(os.path.join(data_dir, "train.pkl"))
    val = load_pickle(os.path.join(data_dir, "val.pkl"))
    test = load_pickle(os.path.join(data_dir, "test.pkl"))
    expr_dict = load_pickle(os.path.join(data_dir, "expr_dict.pkl"))

    embeddings = build_embeddings_from_ast(expr_dict)
    co_usage, conj_to_premises = compute_co_usage(train)

    datasets = {"train": train, "val": val, "test": test}
    for split, pairs in datasets.items():
        cgf_graphs = []
        seen_conj = set()
        for conj, premise, label in pairs:
            if conj not in seen_conj:
                premises_list = conj_to_premises.get(conj, [])
                cgf_graphs.append(build_cgf(conj, premises_list, embeddings, expr_dict, co_usage))
                seen_conj.add(conj)
        with open(os.path.join(out_dir, f"cgf_{split}.pkl"), "wb") as f:
            pickle.dump(cgf_graphs, f)
        print(f"Built {len(cgf_graphs)} CGF graphs ({split} set).")

if __name__ == "__main__":
    main()