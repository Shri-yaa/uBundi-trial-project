# scripts/build_index.py
import os, re, uuid, json, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import markdown

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INDEX_DIR = Path(__file__).resolve().parents[1] / "artifacts"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_md(path: Path) -> str:
    # strip markdown to text-ish
    text = path.read_text(encoding="utf-8", errors="ignore")
    # naive removal of markdown artifacts
    text = re.sub(r"`{1,3}.*?`{1,3}", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text

def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
        if i <= 0: break
    return chunks

def hash_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--out", type=str, default=str(INDEX_DIR / "faiss.index"))
    parser.add_argument("--meta", type=str, default=str(INDEX_DIR / "chunks_meta.json"))
    parser.add_argument("--chunk_size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=180)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = list(data_dir.glob("**/*"))
    files = [f for f in files if f.is_file() and f.suffix.lower() in [".pdf", ".txt", ".md"]]

    if not files:
        print(f"No files found in {data_dir}. Please add documents.")
        return

    model = SentenceTransformer(MODEL_NAME)
    texts, metas = [], []

    for f in files:
        try:
            if f.suffix.lower() == ".pdf":
                raw = read_pdf(f)
            elif f.suffix.lower() == ".md":
                raw = read_md(f)
            else:
                raw = read_txt(f)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue

        for idx, ch in enumerate(chunk_text(raw, args.chunk_size, args.overlap)):
            cid = f"{hash_id(str(f))}-{idx:04d}"
            texts.append(ch)
            metas.append({
                "chunk_id": cid,
                "file": f.name,
                "relpath": str(f.relative_to(data_dir)),
                "char_count": len(ch),
            })

    # Embeddings
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # FAISS index
    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(embs.astype(np.float32))

    faiss.write_index(index, args.out)
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump({"metas": metas}, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(texts)} chunks from {len(files)} files.")
    print(f"Index written to: {args.out}")
    print(f"Metadata written to: {args.meta}")

if __name__ == "__main__":
    main()
