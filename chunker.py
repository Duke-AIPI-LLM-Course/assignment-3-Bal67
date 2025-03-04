import nltk
import json

nltk.download("punkt")

# Load scraped text
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into sentences
sentences = nltk.sent_tokenize(text)

# Chunking function (50 words per chunk)
def chunk_text(sentences, max_tokens=50):
    chunks, current_chunk = [], []
    token_count = 0

    for sentence in sentences:
        tokens = sentence.split()
        if token_count + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, token_count = [], 0
        current_chunk.extend(tokens)
        token_count += len(tokens)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Generate chunks
chunks = chunk_text(sentences)

# Save chunks
with open("chunks.json", "w") as f:
    json.dump(chunks, f)

print(f"✅ Chunking completed: {len(chunks)} chunks created.")
