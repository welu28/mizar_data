import pickle
import os

DATA_DIR = "mizar_pickles/cgf_pickles"
FILES = ["cgf_train.pkl", "cgf_val.pkl", "cgf_test.pkl"]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def verify_cgf_file(file_path):
    print(f"Verifying {file_path}...")
    try:
        data = load_pickle(file_path)
    except Exception as e:
        print(f"Failed to load pickle: {e}")
        return False

    if not isinstance(data, list):
        print(f"Data is not a list, got {type(data)}")
        return False

    for i, graph in enumerate(data):
        if "nodes" not in graph or "edges" not in graph:
            print(f"Graph {i} missing 'nodes' or 'edges'")
            return False

        nodes = graph["nodes"]
        edges = graph["edges"]

        if not isinstance(nodes, dict):
            print(f"Graph {i} nodes is not a dict")
            return False
        if not isinstance(edges, list):
            print(f"Graph {i} edges is not a list")
            return False

        for node_id, node_info in nodes.items():
            if "type" not in node_info:
                print(f"Node {node_id} missing 'type'")
                return False

        for edge in edges:
            if not all(k in edge for k in ("src", "dst", "type", "weight")):
                print(f"Edge {edge} missing keys")
                return False

    print(f"{file_path} verified successfully with {len(data)} graphs.")
    return True

def main():
    all_passed = True
    for file_name in FILES:
        path = os.path.join(DATA_DIR, file_name)
        if not verify_cgf_file(path):
            all_passed = False

    if all_passed:
        print("All CGF files verified successfully")
    else:
        print("Some CGF files failed verification")

if __name__ == "__main__":
    main()