import json
import time
from pathlib import Path

import faiss
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DIM = 384
N_DOCS = 20_000
N_QUERIES = 200
K = 10
M = 32
EF_CONSTRUCTION = 200
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_EMB = Path("embeddings.npy")
CACHE_TEXTS = Path("texts.json")
SWEEP_EF = [8, 16, 32, 64, 128, 256]

_state = {
    "texts": None,
    "emb": None,
    "model": None,
    "exact": None,
    "hnsw": None,
    "query_ids": None,
}


def _load_or_build_corpus():
    if CACHE_EMB.exists() and CACHE_TEXTS.exists():
        emb = np.load(CACHE_EMB).astype(np.float32)
        texts = json.loads(CACHE_TEXTS.read_text())
        if len(texts) == N_DOCS and emb.shape == (N_DOCS, DIM):
            return texts, emb

    ds = load_dataset("fancyzhx/ag_news", split=f"train[:{N_DOCS}]")
    texts = [row["text"].split("\n")[0].strip() or row["text"][:240] for row in ds]
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(CACHE_EMB, emb)
    CACHE_TEXTS.write_text(json.dumps(texts))
    return texts, emb


def init():
    if _state["exact"] is not None:
        return

    texts, emb = _load_or_build_corpus()
    model = SentenceTransformer(MODEL_NAME)

    exact = faiss.IndexFlatIP(DIM)
    exact.add(emb)

    hnsw = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    hnsw.add(emb)

    rng = np.random.default_rng(42)
    query_ids = rng.choice(N_DOCS, size=N_QUERIES, replace=False).astype(np.int64)

    _state.update(
        texts=texts,
        emb=emb,
        model=model,
        exact=exact,
        hnsw=hnsw,
        query_ids=query_ids,
    )


def _search(index, vecs, k=K):
    t0 = time.perf_counter()
    scores, ids = index.search(vecs, k)
    ms = (time.perf_counter() - t0) * 1000.0
    return scores, ids, ms


def benchmark(ef_search: int):
    init()
    _state["hnsw"].hnsw.efSearch = int(ef_search)
    q = _state["emb"][_state["query_ids"]]

    _, exact_ids, exact_ms = _search(_state["exact"], q)
    _, hnsw_ids, hnsw_ms = _search(_state["hnsw"], q)

    recalls = []
    for i in range(N_QUERIES):
        a = set(exact_ids[i].tolist())
        b = set(hnsw_ids[i].tolist())
        recalls.append(len(a & b) / float(K))
    recall = float(np.mean(recalls))
    exact_avg = exact_ms / N_QUERIES
    hnsw_avg = hnsw_ms / N_QUERIES
    return recall, exact_avg, hnsw_avg


def sweep_chart():
    init()
    if _state.get("sweep_fig") is not None:
        return _state["sweep_fig"]
    recalls, latencies = [], []
    for ef in SWEEP_EF:
        recall, _, hnsw_avg = benchmark(ef)
        recalls.append(recall)
        latencies.append(hnsw_avg)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.6))
    ax1.plot(SWEEP_EF, recalls, marker="o", color="#1b4d3e", label="recall@10")
    ax1.set_xlabel("efSearch")
    ax1.set_ylabel("recall@10")
    ax1.set_ylim(0.7, 1.02)
    ax1.set_xticks(SWEEP_EF)
    ax2 = ax1.twinx()
    ax2.plot(SWEEP_EF, latencies, marker="s", color="#b45309", label="HNSW latency (ms)")
    ax2.set_ylabel("HNSW avg latency (ms / query)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower right")
    fig.tight_layout()
    _state["sweep_fig"] = fig
    return fig


def format_hits(scores, ids):
    lines = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx < 0:
            continue
        title = _state["texts"][int(idx)]
        lines.append(f"{rank}. [{score:.3f}] {title}")
    return "\n".join(lines) or "(no hits)"


def run_query(query: str, ef_search: int):
    init()
    query = (query or "").strip()
    if not query:
        query = "stock market rally"

    vec = _state["model"].encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    _state["hnsw"].hnsw.efSearch = int(ef_search)
    exact_scores, exact_ids, exact_ms = _search(_state["exact"], vec)
    hnsw_scores, hnsw_ids, hnsw_ms = _search(_state["hnsw"], vec)

    exact_set = set(exact_ids[0].tolist())
    hnsw_set = set(hnsw_ids[0].tolist())
    q_recall = len(exact_set & hnsw_set) / float(K)

    bench_recall, bench_exact, bench_hnsw = benchmark(ef_search)

    metrics = (
        f"**efSearch** = {int(ef_search)}  \n"
        f"**This query recall@10** = {q_recall:.2f} · exact {exact_ms:.2f} ms · HNSW {hnsw_ms:.2f} ms  \n"
        f"**200-query bench recall@10** = {bench_recall:.3f} · exact {bench_exact:.3f} ms/q · HNSW {bench_hnsw:.3f} ms/q"
    )
    return (
        metrics,
        format_hits(exact_scores[0], exact_ids[0]),
        format_hits(hnsw_scores[0], hnsw_ids[0]),
        sweep_chart(),
    )


def on_slider(ef_search: int, query: str):
    return run_query(query, ef_search)


with gr.Blocks(title="ANN Benchmark Playground") as demo:
    gr.Markdown(
        "# ANN Benchmark Playground\n"
        "Exact FAISS `IndexFlatIP` vs HNSW on 20k `ag_news` titles "
        f"(`{MODEL_NAME}`, cosine via normalized inner product). "
        "Move **efSearch** to trade HNSW recall for latency."
    )
    with gr.Row():
        ef = gr.Slider(8, 256, value=32, step=1, label="efSearch")
        query = gr.Textbox(
            label="Query",
            placeholder="e.g. oil prices rise in asia",
            value="stock market rally",
        )
    metrics = gr.Markdown()
    with gr.Row():
        exact_out = gr.Textbox(label="Exact (FlatIP) top-10", lines=12)
        hnsw_out = gr.Textbox(label="HNSW top-10", lines=12)
    chart = gr.Plot(label="recall@10 vs HNSW latency across efSearch")
    run_btn = gr.Button("Search / refresh bench", variant="primary")

    inputs = [query, ef]
    outputs = [metrics, exact_out, hnsw_out, chart]
    demo.load(run_query, inputs, outputs)
    run_btn.click(run_query, inputs, outputs)
    ef.release(on_slider, [ef, query], outputs)
    query.submit(run_query, inputs, outputs)


if __name__ == "__main__":
    demo.launch()
