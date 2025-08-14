import os
import json
import pickle
import hashlib
import chromadb
import pandas as pd
from tqdm import tqdm
from pydantic import PrivateAttr
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings



class CsvRAGTool(BaseTool):
    name: str = "CsvRAGTool"
    description: str = (
        "Indexes a CSV into ChromaDB and caches row embeddings in .pkl. "
        "Only re-embeds changed rows based on row-level hashing."
    )

    _file_path: str = PrivateAttr()
    _collection_name: str = PrivateAttr()
    _embedding_dir: str = PrivateAttr(default="chroma_db")
    _pkl_cache_path: str = PrivateAttr()

    def __init__(self, file_path: str, collection_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self._file_path = file_path
        self._collection_name = collection_name or self._derive_name()
        self._pkl_cache_path = os.path.join(
            self._embedding_dir, f"{self._collection_name}.pkl"
        )
        self._maybe_index_csv()

    def _derive_name(self):
        return (
            os.path.splitext(os.path.basename(self._file_path))[0]
            .lower()
            .replace(" ", "_")
        )

    def _hash_row(self, row: str) -> str:
        return hashlib.md5(row.encode("utf-8")).hexdigest()

    def _maybe_index_csv(self):
        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"CSV file not found: {self._file_path}")
        os.makedirs(self._embedding_dir, exist_ok=True)

        df = pd.read_csv(self._file_path, dtype=str).fillna("")
        texts = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
        metadata = df.to_dict(orient="records")

        old_cache = {}
        if os.path.exists(self._pkl_cache_path):
            with open(self._pkl_cache_path, "rb") as f:
                old_cache = pickle.load(f)

        old_row_hashes = old_cache.get("row_hashes", {})
        old_embeddings = old_cache.get("embeddings", {})
        old_texts = old_cache.get("texts", {})
        old_metadata = old_cache.get("metadata", {})

        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        new_embeddings = {}
        new_row_hashes = {}
        new_texts = {}
        new_metadata = {}

        print("[CsvRAGTool] Checking for updated rows...")

        to_embed = []
        embed_indices = []

        for idx, row_text in enumerate(texts):
            row_hash = self._hash_row(row_text)
            row_id = f"row{idx}"
            new_row_hashes[row_id] = row_hash

            if old_row_hashes.get(row_id) == row_hash:
                new_embeddings[row_id] = old_embeddings[row_id]
                new_texts[row_id] = old_texts[row_id]
                new_metadata[row_id] = old_metadata[row_id]
            else:
                to_embed.append(row_text)
                embed_indices.append(row_id)
                new_texts[row_id] = row_text
                new_metadata[row_id] = metadata[idx]

      
        if to_embed:
            print(f"[CsvRAGTool] Re-embedding {len(to_embed)} changed rows...")
            batch_size = 500
            for i in tqdm(
                range(0, len(to_embed), batch_size), desc="Embedding updated rows"
            ):
                batch = to_embed[i : i + batch_size]
                embedded = embedding_model.embed_documents(batch)
                for j, row_id in enumerate(embed_indices[i : i + batch_size]):
                    new_embeddings[row_id] = embedded[j]
        else:
            print("[CsvRAGTool] No changes detected. Skipping re-embedding.")

        with open(self._pkl_cache_path, "wb") as f:
            pickle.dump(
                {
                    "row_hashes": new_row_hashes,
                    "embeddings": new_embeddings,
                    "texts": new_texts,
                    "metadata": new_metadata,
                },
                f,
            )

        print("[CsvRAGTool] Rebuilding ChromaDB collection...")
        client = chromadb.PersistentClient(
            path=os.path.join(self._embedding_dir, "chroma")
        )

        try:
            client.delete_collection(self._collection_name)
        except Exception:
            pass

        collection = client.create_collection(self._collection_name)

        collection.add(
            documents=[new_texts[row_id] for row_id in new_texts],
            embeddings=[new_embeddings[row_id] for row_id in new_embeddings],
            metadatas=[new_metadata[row_id] for row_id in new_metadata],
            ids=list(new_texts.keys()),
        )

    def _run(self, query: str) -> str:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        query_embedding = embedding_model.embed_query(query)

        client = chromadb.PersistentClient(
            path=os.path.join(self._embedding_dir, "chroma")
        )
        collection = client.get_collection(self._collection_name)

        results = collection.query(query_embeddings=[query_embedding], n_results=500)
        documents = results.get("documents", [[]])[0]

        if not documents:
            return "No relevant rows found."

        return "\n\n".join(documents)
