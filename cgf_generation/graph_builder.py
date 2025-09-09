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

def build_ast_edges(expr):
    edges = []
    return edges

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

def build_cgf(conjecture, premises, embeddings, co_usage):
    nodes = {}
    edges = []
    nodes[conjecture] = embeddings[conjecture]
    for p in premises:
        nodes[p] = embeddings[p]
        ast_edges = build_ast_edges(p)
        edges.extend(ast_edges)
        edges.append((conjecture, p))
    for (p1, p2) in combinations(premises, 2):
        if (p1, p2) in co_usage:
            edges.append((p1, p2))
            edges.append((p2, p1))
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
                cgf_graphs.append(build_cgf(conj, premises, embeddings, co_usage))
                seen_conj.add(conj)
        with open(os.path.join(out_dir, f"cgf_{split}.pkl"), "wb") as f:
            pickle.dump(cgf_graphs, f)
        print(f"Built {len(cgf_graphs)} CGF graphs ({split} set).")

if __name__ == "__main__":
    main()