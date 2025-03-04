import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load embeddings
with open("database.json", "r") as f:
    db = json.load(f)

chunks = db["chunks"]
chunk_vectors = [np.array(vec) for vec in db["vectors"]]

# Cosine similarity function
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

# Retrieve best chunk
def retrieve_best_chunk(query):
    query_embedding = embed_model.encode(query)
    similarities = [cosine_similarity(query_embedding, vec) for vec in chunk_vectors]
    best_idx = np.argmax(similarities)
    return chunks[best_idx]
