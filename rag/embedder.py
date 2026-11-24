"""Embed chunks and store in ChromaDB."""

from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PATH,
    CHUNK_MAX_CHARS,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    COLLECTION_NAME,
    EMBEDDING_MAX_INPUT_CHARS,
    EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL,
    TABLE_EXTRACT_ENABLED,
)
from rag.embeddings import get_embedding_backend, hosted_configured
from rag.index_manifest import verify_manifest, write_manifest
from rag.provenance import is_allowed

BASE_DIR = Path(__file__).parent.parent
MANIFEST_NAME = "index_manifest.json"
_WARNED_MANIFESTS = set()


class PolicyVectorStore:
    def __init__(self):
        Path(CHROMA_PATH).parent.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        # The hosted endpoint returns the same vectors as this model, so the
        # index stays valid either way. Skip loading the local weights when
        # the API is configured; Chroma accepts explicit embeddings.
        self.embed_fn = None
        if not hosted_configured():
            self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self._encoder = get_embedding_backend(self.embed_fn)
        self._query_cache = {}
        self._catalog_cache = None
        self.manifest_problems = self._check_manifest()

    def _manifest_path(self) -> Path:
        return Path(CHROMA_PATH) / MANIFEST_NAME

    def _check_manifest(self) -> List[str]:
        """Compare the stored index contract with the current config."""
        path = self._manifest_path()
        problems = verify_manifest(
            path,
            model=EMBEDDING_MODEL,
            dim=self._encoder.dim,
        )
        key = (str(path), tuple(problems))
        if problems and key not in _WARNED_MANIFESTS:
            _WARNED_MANIFESTS.add(key)
            for problem in problems:
                print(f"Index manifest: {problem}")
        return problems

    def _chunk_params(self) -> Dict:
        return {
            "chunk_max_tokens": CHUNK_MAX_TOKENS,
            "chunk_overlap_tokens": CHUNK_OVERLAP,
            "chunk_target_chars": CHUNK_TARGET_CHARS,
            "chunk_max_chars": CHUNK_MAX_CHARS,
            "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
            "embed_max_input_chars": EMBEDDING_MAX_INPUT_CHARS,
            "embed_max_input_tokens": EMBEDDING_MAX_INPUT_TOKENS,
            "table_extract": TABLE_EXTRACT_ENABLED,
        }

    def _write_manifest(self):
        try:
            write_manifest(
                self._manifest_path(),
                model=EMBEDDING_MODEL,
                dim=self._encoder.dim,
                chunks=self.collection.count(),
                params=self._chunk_params(),
                extra={
                    "backend": self._encoder.name,
                    "collection": COLLECTION_NAME,
                },
            )
            self.manifest_problems = []
        except OSError as exc:
            print(f"Index manifest not written: {exc}")
    
    @staticmethod
    def _dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
        """Drop chunks whose content hash repeats inside one batch."""
        seen = set()
        unique = []
        for chunk in chunks:
            chunk_id = chunk["metadata"].get("chunk_id")
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            unique.append(chunk)
        return unique

    @staticmethod
    def _chunk_ids(chunks: List[Dict]) -> List[str]:
        ids = []
        for index, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            chunk_id = meta.get("chunk_id")
            ids.append(f"{meta['source']}_{chunk_id or index}")
        return ids

    def add_chunks(self, chunks: List[Dict]):
        chunks = self._dedupe_chunks(chunks)
        if not chunks:
            print("No chunks to add.")
            return
        self._upsert(chunks)

    def _upsert(self, chunks: List[Dict]):
        ids = self._chunk_ids(chunks)
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self._encoder.encode(texts).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._catalog_cache = None
        self._write_manifest()
        print(f"Upserted {len(chunks)} chunks into ChromaDB")

    def replace_source_chunks(self, chunks: List[Dict]):
        """Replace the chunks belonging to exactly one source document.

        Documents that share a `source` key (brochure vs CIS, a re-scrape of
        the same product) must not delete each other, so removal is scoped to
        this file: its relative path when known, otherwise its file name and
        insurer. New chunks are embedded before the old ones are deleted, so
        an embedding failure leaves the previous index intact.
        """
        if not chunks:
            print("No chunks to replace.")
            return

        sources = {chunk["metadata"]["source"] for chunk in chunks}
        if len(sources) != 1:
            raise ValueError("Replacement batch must contain exactly one source.")

        chunks = self._dedupe_chunks(chunks)
        ids = self._chunk_ids(chunks)
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self._encoder.encode(texts).tolist()

        sample = chunks[0]["metadata"]
        source_rel = sample.get("source_rel")
        source_file = sample.get("source_file")
        insurer = sample.get("insurer")
        if source_rel:
            self.collection.delete(where={"source_rel": source_rel})
        if source_file and insurer:
            self.collection.delete(
                where={
                    "$and": [
                        {"source_file": Path(source_file).name},
                        {"insurer": insurer},
                    ]
                }
            )
        elif not source_rel:
            self.collection.delete(where={"source": next(iter(sources))})
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._catalog_cache = None
        self._write_manifest()
        print(f"Upserted {len(chunks)} chunks into ChromaDB")

    def _embed_query(self, query: str):
        vector = self._query_cache.get(query)
        if vector is None:
            vector = self._encoder.encode([query])[0]
            if len(self._query_cache) >= 128:
                self._query_cache.clear()
            self._query_cache[query] = vector
        return vector
    
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
            query_embeddings=[self._embed_query(query).tolist()],
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
        return {
            "total_chunks": self.collection.count(),
            "embedding_backend": self._encoder.name,
            "manifest_problems": list(self.manifest_problems),
        }

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
        batch_size = 2000
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
                if not is_allowed(metadata):
                    continue
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
        """Return indexed sources grouped for the Streamlit knowledge summary.

        Chroma is authoritative for what can actually be quoted. Chunks without
        a ledgered source file are skipped, so a stale demo document cannot
        appear as a loaded policy. The ledger still adds sources that are
        indexed under a different policy key.
        """
        if self._catalog_cache is not None:
            return self._catalog_cache
        from_chroma = self._catalog_from_chroma()
        from_index = self._catalog_from_index_file()
        if not from_chroma:
            self._catalog_cache = from_index
            return self._catalog_cache
        known = {
            (entry["insurer"], entry["policy_name"], entry["document_type"])
            for entry in from_chroma
        }
        for entry in from_index:
            key = (entry["insurer"], entry["policy_name"], entry["document_type"])
            if key not in known:
                known.add(key)
                from_chroma.append(entry)
        from_chroma.sort(
            key=lambda entry: (
                entry["insurer"],
                entry["policy_name"],
                entry["document_type"],
                entry["source_file"],
            )
        )
        self._catalog_cache = from_chroma
        return self._catalog_cache


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