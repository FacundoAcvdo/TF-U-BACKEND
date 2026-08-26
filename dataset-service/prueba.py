import chromadb

# 1. Conéctate al servidor de Chroma en Docker (por defecto usa localhost:8000)
client = chromadb.HttpClient(host="localhost", port=8001)

# 2. Accede a tu colección (usa el nombre exacto de la tuya)
collection = client.get_collection(name="documents")

# 3. Pide una muestra de elementos obligando a incluir los "embeddings"
datos = collection.get(
    limit=2,
    include=["embeddings", "documents", "metadatas"]
)

# 4. Inspecciona los vectores resultantes
for i in range(len(datos["ids"])):
    print(f"📄 Documento: {datos['documents'][i]}")
    # Cortamos el print a los primeros 5 números para no inundar tu terminal
    print(f"📐 Vector (Primeros 5 valores): {datos['embeddings'][i][:5]}...")
    print(f"📊 Dimensión total del vector: {len(datos['embeddings'][i])}\n")
    print(f"🗂️ Metadatos: {datos['metadatas'][i]}\n")
