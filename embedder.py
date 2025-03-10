import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


with open("chunks.json", "r") as f:
    chunks = json.load(f)


chunk_embeddings = [embed_model.encode(chunk).tolist() for chunk in chunks]


db = {"chunks": chunks, "vectors": chunk_embeddings}

with open("database.json", "w") as f:
    json.dump(db, f)

print("Embeddings computed and saved in database.json.")
