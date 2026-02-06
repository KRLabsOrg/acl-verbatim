import argparse

from pymilvus import MilvusClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean (drop) Milvus collections"
    )
    parser.add_argument(
        "--uri",
        default="http://localhost:19530",
        help="Milvus server URI (default: http://localhost:19530)",
    )
    parser.add_argument(
        "--token",
        default="root:Milvus",
        help="Authentication token for Milvus connection (default: root:Milvus)",
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Collection name to drop, or 'ALL' to drop all collections",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = MilvusClient(uri=args.uri, token=args.token)

    if args.collection_name == "ALL":
        collections_to_drop = client.list_collections()
    else:
        collections_to_drop = [args.collection_name]

    for name in collections_to_drop:
        try:
            client.drop_collection(collection_name=name)
            print(f"Dropped {name}")
        except Exception as e:
            print(f"Failed to drop {name}: {e}")

    print("Remaining:", client.list_collections())


if __name__ == "__main__":
    main()
