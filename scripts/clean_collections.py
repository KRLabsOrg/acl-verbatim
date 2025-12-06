from pymilvus import MilvusClient


def main():
    client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

    for name in client.list_collections():
        try:
            client.drop_collection(collection_name=name)
            print(f"Dropped {name}")
        except Exception as e:
            print(f"Failed to drop {name}: {e}")

    print("Remaining:", client.list_collections())


if __name__ == "__main__":
    main()
