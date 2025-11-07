#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch indexer for legal texts (e.g., 民法典 / 刑法)
Steps:
  JSONL -> Embedding -> Chroma
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb


def load_chunks(path: str):
    """Load JSONL chunk file"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"[INFO] loaded {len(chunks)} chunks from {path}")
    return chunks


def sanitize_metadata_remove_none(metas):
    """Remove keys with None value (Chroma不支持None类型)"""
    clean = []
    for md in metas:
        clean_md = {k: v for k, v in md.items() if v is not None}
        clean.append(clean_md)
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, required=True, help="Path to JSONL chunk file")
    parser.add_argument("--persist_dir", type=str, default="./chroma_db", help="Directory for Chroma persistence")
    parser.add_argument("--collection", type=str, default="law_civil", help="Chroma collection name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for embeddings")
    args = parser.parse_args()

    # --- Step 1: Load chunks ---
    items = load_chunks(args.chunks)
    ids = [it["id"] for it in items]
    docs = [it["text"] for it in items]
    metadatas = [{"source": it.get("source"), "article_no": it.get("article_no"), "page": it.get("page")} for it in items]
    metadatas = sanitize_metadata_remove_none(metadatas)

    # --- Step 2: Load embedding model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] using device = {device}")

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
    print("[INFO] loaded Qwen3 Embedding model")

    # --- Step 3: Create Chroma persistent client ---
    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_or_create_collection(args.collection)
    print(f"[INFO] Chroma DB initialized at {args.persist_dir} (collection={args.collection})")

    # --- Step 4: Compute embeddings & insert ---
    all_embeddings = []
    for i in tqdm(range(0, len(docs), args.batch_size), desc="Embedding batches"):
        batch_texts = docs[i:i + args.batch_size]
        with torch.no_grad():
            batch_embeds = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(batch_embeds)

    assert len(all_embeddings) == len(docs), "embedding count mismatch"

    print("[INFO] Inserting into Chroma...")
    collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=all_embeddings)
    print(f"[SUCCESS] Indexed {len(docs)} documents into collection '{args.collection}' ✅")


if __name__ == "__main__":
    main()
