# app/main.py
# -*- coding: utf-8 -*-
import os
import asyncio
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import json
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import logging

app = FastAPI(title="Law RAG (Chroma + Qwen + DeepSeek)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:5500", "http://localhost:8000", "http://tkp666.abrdns.com:8000", "http://tkp666.abrdns.com", "*"],  # 添加域名到CORS白名单
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config from env / defaults ----------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "law_civil")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com")  # or your base
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1500"))
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "120"))
# Django用户服务URL
DJANGO_SERVICE_URL = os.getenv("DJANGO_SERVICE_URL", "http://localhost:8001")  # Django运行在8001端口，使用本地地址


# ---------- Request / Response models ----------
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = DEFAULT_TOP_K
    model: Optional[str] = DEEPSEEK_MODEL  # Add model field for deepseek-chat vs deepseek-reasoner
    user_token: Optional[str] = None  # 添加用户token字段，用于认证


class RetrievalItem(BaseModel):
    source: Optional[str]
    article_no: Optional[str]
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    retrievals: List[RetrievalItem]


# ---------- Global handles (initialized at startup) ----------
MODEL: Optional[SentenceTransformer] = None
CHROMA_CLIENT = None
CHROMA_COLLECTION_OBJ = None
HTTPX_CLIENT: Optional[httpx.AsyncClient] = None
DJANGO_HTTP_CLIENT: Optional[httpx.AsyncClient] = None  # 用于与Django通信


# ---------- Startup / Shutdown ----------
@app.on_event("startup")
async def startup_event():
    global MODEL, CHROMA_CLIENT, CHROMA_COLLECTION_OBJ, HTTPX_CLIENT, DJANGO_HTTP_CLIENT

    # 1. Load embedding model in thread to avoid blocking event loop
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用CPU以避免CUDA内存问题
    print(f"[startup] loading embedding model {QWEN_MODEL} on device {device} ...")
    MODEL = await asyncio.to_thread(SentenceTransformer, QWEN_MODEL, {"device": device})
    # NOTE: If above fails due to kwargs differences, fallback to simple call:
    # MODEL = await asyncio.to_thread(SentenceTransformer, QWEN_MODEL)

    print("[startup] model loaded.")

    # 2. Init Chroma persistent client (local)
    print(f"[startup] opening Chroma at {CHROMA_DIR} ...")
    CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
    CHROMA_COLLECTION_OBJ = CHROMA_CLIENT.get_or_create_collection(name=CHROMA_COLLECTION)
    print(f"[startup] Chroma collection '{CHROMA_COLLECTION}' ready.")

    # 3. HTTPX Async client for DeepSeek
    HTTPX_CLIENT = httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT)
    print("[startup] DeepSeek httpx client ready.")

    # 4. HTTPX Async client for Django user service
    DJANGO_HTTP_CLIENT = httpx.AsyncClient(timeout=30)
    print("[startup] Django httpx client ready.")


@app.on_event("shutdown")
async def shutdown_event():
    global HTTPX_CLIENT, CHROMA_CLIENT, DJANGO_HTTP_CLIENT
    if HTTPX_CLIENT:
        await HTTPX_CLIENT.aclose()
    if DJANGO_HTTP_CLIENT:
        await DJANGO_HTTP_CLIENT.aclose()
    # CHROMA persistent client does not need explicit close
    print("[shutdown] shutdown completed.")


# ---------- Helper functions ----------
async def encode_query_async(text: str):
    """
    Encode query into embedding using MODEL.
    Use prompt_name="query" per Qwen recommendation.
    """
    if MODEL is None:
        raise RuntimeError("Embedding model not loaded")
    # run in thread to avoid blocking
    def _encode():
        return MODEL.encode([text], prompt_name="query", convert_to_numpy=True)[0]
    emb = await asyncio.to_thread(_encode)
    return emb.tolist()


async def retrieve_top_k(query_embedding: List[float], k: int):
    """
    Query Chroma for top-k results. This is IO/CPU bound so run in thread.
    Returns list of (doc, metadata, distance)
    """
    if CHROMA_COLLECTION_OBJ is None:
        raise RuntimeError("Chroma collection not ready")

    def _query():
        return CHROMA_COLLECTION_OBJ.query(query_embeddings=[query_embedding], n_results=k)

    res = await asyncio.to_thread(_query)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    return list(zip(docs, metas, distances))


async def call_deepseek_api(prompt: str, model: str = DEEPSEEK_MODEL) -> str:
    """
    Call DeepSeek chat completions (OpenAI-compatible style).
    """
    api_key = DEEPSEEK_API_KEY
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in environment")

    url = DEEPSEEK_BASE.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是法律检索辅助助手，基于提供的法条准确回答用户问题，必须引用条号与原文片段；不确定时请提示咨询律师。"},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        r = await HTTPX_CLIENT.post(url, headers=headers, json=body)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"DeepSeek API request failed: {e}")

    data = r.json()
    # try to be tolerant with response shape
    # typical: data['choices'][0]['message']['content']
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        # fallback: try other fields
        if "message" in data and isinstance(data["message"], dict):
            content = data["message"].get("content", json.dumps(data, ensure_ascii=False))
        else:
            content = json.dumps(data, ensure_ascii=False)
    return content


async def save_query_history_to_django(user_token: str, question: str, answer: str, retrievals: List[Dict]):
    """
    保存查询历史到Django用户服务
    """
    if not user_token or user_token == "anonymous":
        # 匿名用户，不保存到数据库
        return
    
    try:
        payload = {
            "question": question,
            "answer": answer,
            "source_info": {
                "retrievals": retrievals
            }
        }
        
        # 添加认证头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {user_token}"
        }
        url = f"{DJANGO_SERVICE_URL}/api/auth/save-query/"
        
        response = await DJANGO_HTTP_CLIENT.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print(f"[INFO] Query history saved to Django service for user token: {user_token}")
        else:
            print(f"[ERROR] Failed to save query history to Django service: {response.status_code}, {response.text}")
    
    except Exception as e:
        print(f"[ERROR] Exception while saving query history to Django: {e}")


# ---------- Endpoint ----------
@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")

    top_k = req.top_k or DEFAULT_TOP_K
    # 1) encode query
    try:
        query_emb = await encode_query_async(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}")

    # 2) retrieve top_k
    try:
        hits = await retrieve_top_k(query_emb, top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}")

    # 3) build context (concat retrieved snippets with metadata)
    retrievals = []
    context_parts = []
    for doc, meta, dist in hits:
        score = 1 - dist if isinstance(dist, (int, float)) else None
        src = meta.get("source") if meta else None
        art = meta.get("article_no") if meta else None
        # sanitize doc & metadata
        doc_text = doc if isinstance(doc, str) else str(doc)
        retrievals.append({"source": src, "article_no": art, "text": doc_text, "score": score})
        context_parts.append(f"来源: {src or '未知'} 条号: {art or '无'}\n{doc_text}")

    context = "\n\n".join(context_parts)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "…（已截断）"

    # 4) compose prompt for DeepSeek
    prompt = f"""用户问题：
{question}

检索到的相关法条 / 段落：
{context}

请基于上述法条回答用户问题：
1) 先给出结论（简短明确）
2) 列出支持结论的法条编号与摘录（每条至少一段原文）
3) 若无法确定，明确写"无法确定，请咨询律师"
4) 最后写上"免责声明：本回答仅供参考，非法律意见" """

    # 5) call DeepSeek with the specified model
    selected_model = req.model or DEEPSEEK_MODEL
    try:
        answer = await call_deepseek_api(prompt, model=selected_model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    # 6) Save query history asynchronously, if user is authenticated
    if req.user_token and req.user_token != "anonymous":
        # 在后台保存查询历史，不阻塞主要请求响应
        asyncio.create_task(save_query_history_to_django(req.user_token, question, answer, [
            {"source": r["source"], "article_no": r["article_no"], "text": r["text"], "score": r["score"]}
            for r in retrievals
        ]))

    # 7) return structured response
    return {"answer": answer, "retrievals": retrievals}


# ---------- 获取用户历史记录端点 ----------
@app.get("/history")
async def get_user_history(user_token: str = None):
    """
    从Django服务获取用户查询历史
    注意：这是一个简化的实现
    """
    if not user_token or user_token == "anonymous":
        # 匿名用户，返回空历史
        return {"history": []}
    
    try:
        # 添加认证头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {user_token}"
        }
        url = f"{DJANGO_SERVICE_URL}/api/auth/history/"
        
        response = await DJANGO_HTTP_CLIENT.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"[ERROR] Failed to fetch history from Django service: {response.status_code}, {response.text}")
            return {"history": []}
    
    except Exception as e:
        print(f"[ERROR] Exception while fetching history from Django: {e}")
        return {"history": []}


# ---------- 删除用户历史记录端点 ----------
@app.delete("/history/{history_id}")
async def delete_user_history_delete(history_id: int, user_token: str = None):
    """
    从Django服务删除用户查询历史 (DELETE方法)
    """
    if not user_token or user_token == "anonymous":
        # 匿名用户，无权限删除
        raise HTTPException(status_code=401, detail="未认证用户无权限删除历史记录")

    try:
        # 添加认证头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {user_token}"
        }
        url = f"{DJANGO_SERVICE_URL}/api/auth/history/{history_id}/"

        response = await DJANGO_HTTP_CLIENT.delete(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"[ERROR] Failed to delete history from Django service: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"删除失败: {response.text}")

    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTP error while deleting history from Django: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"删除失败: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Exception while deleting history from Django: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.post("/history/{history_id}/delete")
async def delete_user_history_post(history_id: int, request: Request):
    """
    从Django服务删除用户查询历史 (POST方法，备用)
    """
    # 从请求头中获取认证信息
    authorization_header = request.headers.get("Authorization")
    if not authorization_header or not authorization_header.startswith("Token "):
        raise HTTPException(status_code=401, detail="未提供有效的认证令牌")

    user_token = authorization_header[6:]  # 移除 "Token " 前缀

    if not user_token or user_token == "anonymous":
        # 匿名用户，无权限删除
        raise HTTPException(status_code=401, detail="未认证用户无权限删除历史记录")

    try:
        # 添加认证头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {user_token}"
        }
        url = f"{DJANGO_SERVICE_URL}/api/auth/history/{history_id}/"

        response = await DJANGO_HTTP_CLIENT.delete(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"[ERROR] Failed to delete history from Django service (POST): {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"删除失败: {response.text}")

    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTP error while deleting history from Django: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"删除失败: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Exception while deleting history from Django: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# ---------- simple health check ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- run if main ----------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

frontend_path = Path(__file__).resolve().parents[1] / "frontend"  # adjust if needed
if not frontend_path.exists():
    # 方便调试：如果不存在，FastAPI 仍然可运行
    print(f"[startup] frontend folder not found at {frontend_path}, static not mounted.")
else:
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    print(f"[startup] Mounted frontend at {frontend_path}")