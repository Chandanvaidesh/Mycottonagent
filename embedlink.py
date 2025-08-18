import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# -----------------------------
# Configuration
# -----------------------------
CHUNKS_FILE = Path("marketrate.json")  # single file, not folder
COLLECTION_NAME = "cotton_rate_chunks"
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
# Load chunks from file
# -----------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print(f"Loaded {len(all_chunks)} chunks.")

# -----------------------------
# Generate embeddings and upload in batches
# -----------------------------
BATCH_SIZE = 64

for start_idx in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[start_idx:start_idx + BATCH_SIZE]
    texts = [c for c in batch]  # because it's just plain text list
    embeddings = model.encode(texts, convert_to_numpy=True)

    points = []
    for i, text in enumerate(batch):
        points.append({
            "id": start_idx + i,
            "vector": embeddings[i],
            "payload": {
                "chunk_index": start_idx + i,
                "text": text
            }
        })

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

print("✅ All embeddings stored in Qdrant successfully!")
