import os
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# -----------------------------
# Configuration
# -----------------------------
CHUNKS_FOLDER = Path("schemes_chunks.json")
COLLECTION_NAME = "cotton_scheme_chunks"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# Initialize model and client
# -----------------------------
model = SentenceTransformer(EMBED_MODEL_ID)
client = QdrantClient(host="localhost", port=6333)

# -----------------------------
# Create or reset Qdrant collection
# -----------------------------
VECTOR_SIZE = 384  # embedding size for all-MiniLM-L6-v2

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

# -----------------------------
# Load chunks and prepare embeddings
# -----------------------------
all_chunks = []

for file_name in os.listdir(CHUNKS_FOLDER):
    if file_name.endswith(".json"):
        file_path = CHUNKS_FOLDER / file_name
        with open(file_path, "r", encoding="utf-8") as f:
            chunk_list = json.load(f)
            for idx, chunk_text in enumerate(chunk_list):
                all_chunks.append({
                    "source_file": file_name,
                    "chunk_index": idx,
                    "text": chunk_text
                })

print(f"Loaded {len(all_chunks)} chunks.")

# -----------------------------
# Generate embeddings and upload in batches
# -----------------------------
BATCH_SIZE = 64

for start_idx in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[start_idx:start_idx+BATCH_SIZE]
    texts = [c["text"] for c in batch]
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Prepare batch points
    points = []
    for i, chunk in enumerate(batch):
        points.append({
            "id": start_idx + i,
            "vector": embeddings[i],
            "payload": {
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            }
        })

    # Upsert batch to Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

print("✅ All embeddings stored in Qdrant successfully!")
