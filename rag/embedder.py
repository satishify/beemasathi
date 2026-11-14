"""Embed chunks and store in ChromaDB."""

from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME

BASE_DIR = Path(__file__).parent.parent


class PolicyVectorStore:
    def __init__(self):
        Path(CHROMA_PATH).parent.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[Dict]):
        if not chunks:
            print("No chunks to add.")
            return
            
        ids = [f"{c['metadata']['source']}_{i}" for i, c in enumerate(chunks)]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        print(f"Upserted {len(chunks)} chunks into ChromaDB")

    def replace_source_chunks(self, chunks: List[Dict]):
        """Replace all existing chunks for exactly one source."""
        if not chunks:
            print("No chunks to replace.")
            return

        sources = {chunk["metadata"]["source"] for chunk in chunks}
        if len(sources) != 1:
            raise ValueError("Replacement batch must contain exactly one source.")

        self.collection.delete(where={"source": next(iter(sources))})
        self.add_chunks(chunks)
    
    def search(self, query: str, insurer: str = None, policy_name: str = None, 
               chunk_type: str = None, n_results: int = 5) -> List[Dict]:
        filters = []
        if insurer:
            filters.append({"insurer": insurer})
        if policy_name:
            filters.append({"policy_name": policy_name})
        if chunk_type:
            filters.append({"chunk_type": chunk_type})

        if len(filters) > 1:
            where = {"$and": filters}
        elif filters:
            where = filters[0]
        else:
            where = None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        ids = (results.get("ids") or [[]])[0] or []
        documents = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []
        for i in range(len(ids)):
            formatted.append({
                "text": documents[i],
                "metadata": metadatas[i] or {},
                "distance": distances[i] if i < len(distances) else None,
            })
        return formatted
    
    def get_stats(self) -> Dict:
        return {"total_chunks": self.collection.count()}

    def _catalog_from_index_file(self) -> List[Dict]:
        """Build the catalog from the ingest ledger instead of loading every chunk."""
        index_path = BASE_DIR / "data" / "chunks" / "ingest_index.json"
        if not index_path.exists():
            return []
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        catalog = {}
        for rel, meta in (payload.get("files") or {}).items():
            key = (
                meta.get("insurer", "unknown"),
                meta.get("policy_name", "unknown"),
                meta.get("document_type", "policy_wording"),
                Path(rel).name,
            )
            catalog[key] = catalog.get(key, 0) + int(meta.get("chunks") or 0)
        return self._format_catalog(catalog)

    def _catalog_from_chroma(self) -> List[Dict]:
        """Page through Chroma metadata so SQLite does not hit 'too many SQL variables'."""
        catalog = {}
        batch_size = 200
        offset = 0
        while True:
            result = self.collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            metadatas = result.get("metadatas") or []
            if not metadatas:
                break
            for metadata in metadatas:
                metadata = metadata or {}
                key = (
                    metadata.get("insurer", "unknown"),
                    metadata.get("policy_name", "unknown"),
                    metadata.get("document_type", "policy_wording"),
                    metadata.get("source_file", metadata.get("source", "unknown")),
                )
                catalog[key] = catalog.get(key, 0) + 1
            offset += len(metadatas)
            if len(metadatas) < batch_size:
                break
        return self._format_catalog(catalog)

    def _format_catalog(self, catalog: Dict) -> List[Dict]:
        return [
            {
                "insurer": key[0],
                "policy_name": key[1],
                "document_type": key[2],
                "source_file": key[3],
                "chunks": count,
            }
            for key, count in sorted(catalog.items())
        ]

    def get_catalog(self) -> List[Dict]:
        """Return indexed sources grouped for the Streamlit knowledge summary."""
        from_index = self._catalog_from_index_file()
        if from_index:
            return from_index
        return self._catalog_from_chroma()


def index_all_chunks():
    """Read unified chunks and index them."""
    chunks_file = BASE_DIR / "data" / "chunks" / "all_policies.json"
    
    if not chunks_file.exists():
        print(f"❌ No chunks file found at {chunks_file}")
        print("Run: python rag/chunker.py")
        return
    
    chunks = json.loads(chunks_file.read_text())
    store = PolicyVectorStore()
    store.add_chunks(chunks)
    print(f"DB Stats: {store.get_stats()}")


if __name__ == "__main__":
    index_all_chunks()