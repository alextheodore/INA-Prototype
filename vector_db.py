import chromadb
from chromadb.config import Settings

# Setup ChromaDB lokal
client = chromadb.PersistentClient(path="./chroma_db")

# Buat collection untuk pengetahuan UMKM
collection = client.get_or_create_collection(
    name="umkm_knowledge",
    metadata={"description": "Knowledge base UMKM Indonesia"}
)

# Tambahkan data contoh
collection.add(
    documents=[
        "Cara membuat business plan untuk UMKM",
        "Registrasi perizinan UMKM di Indonesia",
        "Strategi pemasaran digital untuk UKM",
        "Cara mengajukan KUR (Kredit Usaha Rakyat)",
        "Tips manajemen keuangan usaha kecil"
    ],
    metadatas=[
        {"category": "business_plan"},
        {"category": "legal"},
        {"category": "marketing"},
        {"category": "funding"},
        {"category": "finance"}
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)