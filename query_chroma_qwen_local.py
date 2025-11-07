#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_and_answer_deepseek.py

功能：
- 从用户问题开始
- 使用本地 embedding 模型 + 本地向量库 (Chroma) 做检索
- 将 top-k 条文拼入 prompt
- 调用 DeepSeek API 生成答案
- 打印 / 返回答案

配置：
- 设置环境变量 DEEPSEEK_API_KEY
- 参数: --persist_dir, --collection, --top_k, --model (deepseek 模型名称), --max_context_chars
"""

import os
import argparse
import torch
from sentence_transformers import SentenceTransformer
import chromadb

import httpx
import json

def load_chroma_collection(persist_dir, collection_name):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    return collection

def retrieve_chunks(collection, embedding_model, query, top_k):
    # Encode query to embedding
    with torch.no_grad():
        query_emb = embedding_model.encode([query], prompt_name="query", convert_to_numpy=True)[0]
    results = collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    scores = results["distances"][0]
    return list(zip(docs, metas, scores))

def call_deepseek(api_key, model, prompt, system_prompt="你是法律问答助手."):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    resp = httpx.post(url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # 默认取第一 choice 的 message.content
    answer = data["choices"][0]["message"]["content"]
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", type=str, default="./chroma_db", help="Chroma persist directory")
    parser.add_argument("--collection", type=str, default="law_civil", help="Chroma collection name")
    parser.add_argument("--top_k", type=int, default=3, help="Top k chunks to retrieve")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model to call (deepseek-chat or deepseek-reasoner)")
    parser.add_argument("--max_context_chars", type=int, default=1500, help="Maximum characters of context to include")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for embedding model")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("Please set environment variable DEEPSEEK_API_KEY")

    # Load embedding model
    print(f"[INFO] Loading embedding model {args.embedding_model} on device {args.device}")
    embed_model = SentenceTransformer(args.embedding_model, device=args.device)

    # Load Chroma collection
    print(f"[INFO] Loading Chroma collection from {args.persist_dir}, collection={args.collection}")
    col = load_chroma_collection(args.persist_dir, args.collection)

    # Get user query
    user_query = input("请输入你的法律问题：\n> ").strip()
    if not user_query:
        print("退出：未输入问题。")
        return

    # Retrieve chunks
    print("[INFO] Retrieving relevant chunks ...")
    hits = retrieve_chunks(col, embed_model, user_query, args.top_k)

    # Build context prompt from hits
    context_parts = []
    for idx, (doc, meta, score) in enumerate(hits):
        src = meta.get("source", "")
        art = meta.get("article_no", "")
        context_parts.append(f"来源: {src} 条号: {art}\n正文: {doc}")

    context = "\n\n".join(context_parts)
    if len(context) > args.max_context_chars:
        context = context[:args.max_context_chars] + "…（已截断）"

    prompt = f"""用户问题：
{user_query}

相关法条／段落：
{context}

请基于上述法条回答用户问题。请先给出结论，然后引用来源的条号与原文摘录。如无法确定，请说明「无法确定，请咨询律师」。"""
    print("[INFO] Prompt:", prompt)

    # Call DeepSeek
    print(f"[INFO] Calling DeepSeek API with model {args.model} ...")
    answer = call_deepseek(api_key, args.model, prompt)

    print("\n=== 回答 ===")
    print(answer)

if __name__ == "__main__":
    main()
