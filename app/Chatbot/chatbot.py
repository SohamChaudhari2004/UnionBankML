import chromadb
from sentence_transformers import SentenceTransformer


client = chromadb.Client()

collection = client.create_collection(name="loan_data")

# Read text file
with open("loans.txt", "r") as file:
    text = file.read()

# Split into chunks (optional)
chunk_size = 500  # Adjust chunk size as needed
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = embedder.encode(chunks)

# Store embeddings in ChromaDB
for i, chunk in enumerate(chunks):
    collection.add(
        ids=[f"chunk-{i}"],
        embeddings=[embeddings[i]],
        metadatas=[{"source": "loans.txt"}],
        documents=[chunk]
    )


query = "What are the options for low-interest loans?"
query_embedding = embedder.encode(query)

# Query ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print(results)
