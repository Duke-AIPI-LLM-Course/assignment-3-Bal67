import numpy as np
import json
from sentence_transformers import SentenceTransformer


embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


with open("database.json", "r") as f:
    db = json.load(f)

chunks = db["chunks"]
chunk_vectors = [np.array(vec) for vec in db["vectors"]]


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)


def retrieve_best_chunks(query, top_k=5):
    query_embedding = embed_model.encode(query)
    similarities = [cosine_similarity(query_embedding, vec) for vec in chunk_vectors]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    retrieved_chunks = [chunks[i] for i in top_indices]

    
    retrieved_chunks = [chunk for chunk in retrieved_chunks if len(chunk.split()) > 20]

    return " ".join(retrieved_chunks)
