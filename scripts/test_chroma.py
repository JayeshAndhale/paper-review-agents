import chromadb
client = chromadb.PersistentClient(path="./data/chroma")
for col in client.list_collections():
    c = client.get_collection(col.name)
    print(col.name, c.count())