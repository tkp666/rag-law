#!/usr/bin/env python3
"""
pdf_chunker.py

功能：
- 遍历 input_dir 下所有 PDF
- 使用 pdfplumber 提取每页文本；若某页文本太短则用 OCR (pytesseract + pdf2image)
- 尝试用正则按“第...条”切分为条文 chunk（支持中文数字与阿拉伯数字）
- 若未匹配到条文，则按字符滑窗 chunk（可配置 chunk_size / overlap）
- 输出每个 chunk 到 JSONL 文件（每行一个 JSON）
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm
import regex as re

# ---------- 配置默认值 ----------
CHUNK_SIZE_DEF = 600
CHUNK_OVERLAP_DEF = 120
MIN_TEXT_LEN_OCR = 50

# 正则：匹配“第...条” (支持中文数字与阿拉伯数字)
ARTICLE_RE = re.compile(r"(第[\p{Han}0-9〇一二三四五六七八九十百零]+条)[\s\S]*?(?=(第[\p{Han}0-9〇一二三四五六七八九十百零]+条)|\Z)")

# ---------- 工具函数 ----------
def extract_pages_pdfplumber(pdf_path: str) -> List[str]:
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                pages.append(txt.strip())
    except Exception as e:
        print(f"[WARN] pdfplumber failed for {pdf_path}: {e}", file=sys.stderr)
    return pages

def ocr_pdf_with_tesseract(pdf_path: str, dpi:int=200) -> List[str]:
    pages_text = []
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        for img in images:
            txt = pytesseract.image_to_string(img, lang='chi_sim+eng')
            pages_text.append(txt.strip())
    except Exception as e:
        print(f"[ERROR] OCR failed for {pdf_path}: {e}", file=sys.stderr)
    return pages_text

def split_by_article(full_text: str) -> List[Tuple[Optional[str], str]]:
    items = []
    for m in ARTICLE_RE.finditer(full_text):
        header = m.group(1)
        body = m.group(0)
        body_text = body[len(header):].strip()
        items.append((header, body_text))
    return items

def sliding_chunks(text: str, chunk_size:int=CHUNK_SIZE_DEF, overlap:int=CHUNK_OVERLAP_DEF) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks

# ---------- 处理单个 PDF ----------
def process_pdf(pdf_path: str, chunk_size:int, overlap:int, min_text_len_ocr:int):
    filename = os.path.basename(pdf_path)
    pages = extract_pages_pdfplumber(pdf_path)
    # 标记需要 OCR 的页
    need_ocr = [not p or len(p) < min_text_len_ocr for p in pages]
    if any(need_ocr):
        ocr_pages = ocr_pdf_with_tesseract(pdf_path)
        # 如果长度相等就替换被标记的页，否则只替换空页
        if len(ocr_pages) == len(pages):
            for i, n in enumerate(need_ocr):
                if n:
                    pages[i] = ocr_pages[i]
        else:
            for i, p in enumerate(pages):
                if (not p or len(p) < min_text_len_ocr) and i < len(ocr_pages):
                    pages[i] = ocr_pages[i]

    full_text = "\n\n".join([p for p in pages if p])
    article_items = split_by_article(full_text)
    chunks = []  # 每项为 dict: {id, source, article_no, page, text}
    if article_items:
        for idx, (article_no, art_text) in enumerate(article_items):
            if not art_text or len(art_text.strip()) < 10:
                continue
            if len(art_text) > chunk_size:
                subs = sliding_chunks(art_text, chunk_size, overlap)
                for sidx, sc in enumerate(subs):
                    cid = f"{filename}::article_{idx}::sub_{sidx}"
                    chunks.append({"id": cid, "source": filename, "article_no": article_no, "page": None, "text": sc.strip()})
            else:
                cid = f"{filename}::article_{idx}"
                chunks.append({"id": cid, "source": filename, "article_no": article_no, "page": None, "text": art_text.strip()})
    else:
        # fallback: per-page chunk
        for pidx, ptxt in enumerate(pages):
            if not ptxt or len(ptxt.strip()) < 5:
                continue
            subs = sliding_chunks(ptxt, chunk_size, overlap)
            for sidx, sc in enumerate(subs):
                cid = f"{filename}::page_{pidx}::chunk_{sidx}"
                chunks.append({"id": cid, "source": filename, "article_no": None, "page": pidx, "text": sc.strip()})
    return chunks

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", required=True, help="PDF input directory")
    parser.add_argument("--out_dir", "-o", default="./chunks_out", help="output dir for jsonl chunks")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE_DEF)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_DEF)
    parser.add_argument("--min_text_len_ocr", type=int, default=MIN_TEXT_LEN_OCR)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted([p for p in input_dir.glob("**/*.pdf")])
    if not pdf_files:
        print(f"[ERROR] no pdf under {input_dir}", file=sys.stderr)
        sys.exit(2)

    out_jsonl = out_dir / "chunks.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for pdf in tqdm(pdf_files, desc="pdfs"):
            try:
                chunks = process_pdf(str(pdf), args.chunk_size, args.overlap, args.min_text_len_ocr)
                for c in chunks:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[ERROR] failed process {pdf}: {e}", file=sys.stderr)

    print(f"[DONE] wrote chunks to {out_jsonl} ({out_jsonl.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
