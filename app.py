# app.py (fixed for OpenAI v1 client)
import os, json, time, csv
from pathlib import Path
from typing import List, Dict
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI   # v1 client import

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

INDEX_PATH = ART_DIR / "faiss.index"
META_PATH = ART_DIR / "chunks_meta.json"
PROMPT_PATH = Path("prompts/system_prompt.txt")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        st.error("No index found. Please add files to data/ and run: `python3 scripts/build_index.py`")
        st.stop()
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metas = json.load(f)["metas"]
    return index, metas

@st.cache_data(show_spinner=False)
def read_system_prompt():
    return Path(PROMPT_PATH).read_text(encoding="utf-8")

def embed(q: str, model: SentenceTransformer) -> np.ndarray:
    v = model.encode([q], normalize_embeddings=True)
    return v.astype(np.float32)

def retrieve(query: str, k: int, emb_model, index, metas):
    qv = embed(query, emb_model)
    sims, idxs = index.search(qv, k)
    idxs = idxs[0].tolist()
    sims = sims[0].tolist()
    rows = []
    for rank, (i, s) in enumerate(zip(idxs, sims)):
        m = metas[i]
        rows.append({**m, "rank": rank + 1, "score": round(float(s), 4)})
    return rows

def format_context(rows: List[Dict]):
    return "\n".join([f"[{r['rank']}] (score={r['score']}) {r['file']} :: chunk {r['chunk_id']}" for r in rows])

def call_llm(api_key: str, system_prompt: str, mode: str, question: str, context_note: str):
    client = OpenAI(api_key=api_key)  # v1 client instance

    mode_map = {
        "Interview": "Use Interview mode.",
        "Personal Story": "Use Personal storytelling mode.",
        "Fast Facts": "Use Fast facts mode.",
        "Humble Brag": "Use Humble brag mode."
    }
    directive = mode_map.get(mode, "Use Interview mode.")
    sys = f"{system_prompt}\n\nCurrent mode: {directive}\n\nUse the following retrieval notes when helpful:\n{context_note}\n"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content

def log_interaction(row: Dict):
    log_path = ART_DIR / "log.csv"
    exists = log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def main():
    st.set_page_config(page_title="Shriyaa's Personal Codex Agent", layout="wide")

    st.title("Shriyaa's Personal Codex Agent")
    st.caption("A personal AI that answers in my voice using my own docs!")

    # Sidebar
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Choose one of my voices!", ["Interview", "Personal Story", "Fast Facts", "Humble Brag"])
    top_k = st.sidebar.slider(
        "Retrieved chunks (k)",
        min_value=1,
        max_value=8,
        value=4,
        help="Controls how many document snippets are retrieved from your data and passed to the model as context."
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input("OpenAI API Key", value=api_key, type="password")
    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API key to run the agent.")
        st.stop()

    emb_model = load_embedder()
    index, metas = load_index()
    system_prompt = read_system_prompt()

    st.subheader("Ask your question")
    q = st.text_input("e.g., What projects are you most proud of?")
    go = st.button("Ask")

    if go and q.strip():
        with st.spinner("Thinking..."):
            rows = retrieve(q, top_k, emb_model, index, metas)
            context_note = format_context(rows)
            answer = call_llm(api_key, system_prompt, mode, q, context_note)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Retrieved context (metadata)")
        st.code(context_note, language="text")

        log_interaction({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": q,
            "mode": mode,
            "retrieval": context_note,
            "answer_preview": (answer[:200] + "…") if len(answer) > 200 else answer
        })

        st.success("Interaction logged to artifacts/log.csv")

    st.markdown("---")
    st.markdown("**Tip:** Update your docs in `data/` and re-run `python3 scripts/build_index.py` to refresh the knowledge base.")

if __name__ == "__main__":
    main()
