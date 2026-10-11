import pickle


GRAPH_FILE = "data/evaluation/legal_graph.gpickle"


def main():

    print("=" * 80)
    print("LEGAL GRAPH TEST")
    print("=" * 80)

    with open(GRAPH_FILE, "rb") as file:
        graph = pickle.load(file)

    print(f"\nNodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    node_types = {}

    for node, data in graph.nodes(data=True):

        node_type = data.get("node_type", "unknown")

        if node_type not in node_types:
            node_types[node_type] = 0

        node_types[node_type] += 1

    print("\nNode types:")

    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")

    print("\nExample nodes:")

    count = 0

    for node, data in graph.nodes(data=True):

        print(
            f"  {node} | "
            f"{data.get('node_type', '')}"
        )

        count += 1

        if count >= 10:
            break

    print("\nExample edges:")

    count = 0

    for source, target, data in graph.edges(data=True):

        print(
            f"  {source}"
            f" -> "
            f"{target}"
            f" | "
            f"{data.get('relationship', '')}"
        )

        count += 1

        if count >= 10:
            break


if __name__ == "__main__":
    main()

