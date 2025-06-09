import chromadb
from chromadb.config import Settings

"""file encapsulates the ChromaDB client and collection setup 
so that other parts of the system can easily retrieve the "radix_docs" collection 
without rewriting the connection code."""

def get_chroma_client():
    return chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(
            allow_reset=True, anonymized_telemetry=False, is_persistent=True
        ),
    )


def init_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="radix_docs_chroma",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None, # Use default embedding function
    )