import numpy as np
from sentence_transformers import SentenceTransformer
import json

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("chunks.json", "r") as f:
    chunks = json.load(f)

chunk_embeddings = [embed_model.encode(chunk) for chunk in chunks]

db = {"chunks": chunks, "vectors": [vec.tolist() for vec in chunk_embeddings]}

with open("database.json", "w") as f:
    json.dump(db, f)

print("Embeddings computed and stored.")
