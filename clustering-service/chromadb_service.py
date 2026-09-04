import chromadb
import numpy as np

def get_dataset_embeddings(dataset_id: str, user_id: str):
    client = chromadb.HttpClient(host="chroma-service", port=8000)
    collection = client.get_collection(name="documents")

    resultados = collection.get(
        where= {
            "$and": [
                {"user_id": user_id},
                {"dataset_id": dataset_id }
            ]
        },
        include=["embeddings"]
    )

    return np.array(resultados["embeddings"], dtype=np.float32)