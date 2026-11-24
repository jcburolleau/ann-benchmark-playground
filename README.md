---
title: ANN Benchmark Playground
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
short_description: Exact vs HNSW recall@10 and latency on real embeddings
python_version: "3.12"
startup_duration_timeout: 30m
---

# ANN Benchmark Playground

Compare exact FAISS inner-product search (`IndexFlatIP`) against HNSW on 20k `fancyzhx/ag_news` titles embedded with `all-MiniLM-L6-v2`.

Move **efSearch** to see recall@10 vs latency live. Embeddings are built on first run and cached on disk (not committed).
