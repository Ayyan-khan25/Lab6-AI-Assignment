"""Task 3: evaluate chunking/retrieval configurations and export retrieval_log.csv."""
import csv, json, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from task2_chunks import make_chunks

def evaluate(scores, chunks, items):
    hits, reciprocal = [], []
    for row, q in zip(scores, items):
        if q["doc"] is None:
            continue
        top = np.argsort(-row, kind="stable")[:3]
        good = [rank for rank, i in enumerate(top, 1)
                if chunks[i]["doc"] == q["doc"]
                and q["evidence"] in chunks[i]["text"]]
        hits.append(bool(good))
        reciprocal.append(1 / good[0] if good else 0)
    if not hits:
        return None, None
    return float(np.mean(hits)), float(np.mean(reciprocal))

def main():
    with open("corpus.json", encoding="utf-8") as f:
        docs = json.load(f)
    with open("queries.json", encoding="utf-8") as f:
        items = json.load(f)
    assert len(docs) >= 5
    answerable = [q for q in items if q["doc"] is not None]
    unanswerable = [q for q in items if q["doc"] is None]
    assert len(answerable) >= 8 and len(unanswerable) >= 2
    for q in answerable:
        assert q["evidence"] and q["evidence"] in docs[q["doc"]]

    # Load the model separately from embedding/encoding timing.
    model = None
    model_load_seconds = None
    try:
        from sentence_transformers import SentenceTransformer
        t = time.perf_counter()
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        model_load_seconds = time.perf_counter() - t
        print(f"Model load seconds: {model_load_seconds:.3f}")
    except Exception as exc:
        print("Embedding model unavailable:", exc)
        print("Continuing with TF-IDF only. Install/download the model to complete embedding comparisons.")

    query_texts = [q["query"] for q in items]
    logs = []
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(docs, strategy)
        texts = [c["text"] for c in chunks]
        print(f"\nStrategy={strategy}; chunk_count={len(chunks)}")
        configurations = ["tfidf"] + (["embedding"] if model is not None else [])
        for method in configurations:
            t = time.perf_counter()
            if method == "tfidf":
                vectorizer = TfidfVectorizer()
                C = vectorizer.fit_transform(texts)
                Q = vectorizer.transform(query_texts)
                encode_seconds = time.perf_counter() - t
                t = time.perf_counter()
                scores = (Q @ C.T).toarray()
                search_seconds = time.perf_counter() - t
            else:
                token_lengths = [
                    len(model.tokenizer(x, truncation=False)["input_ids"])
                    for x in texts + query_texts
                ]
                if max(token_lengths) > model.max_seq_length:
                    raise ValueError("Model token limit exceeded; rechunk or shorten input.")
                C = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                Q = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)
                encode_seconds = time.perf_counter() - t
                t = time.perf_counter()
                scores = Q @ C.T
                search_seconds = time.perf_counter() - t

            for split in ["dev", "test"]:
                indices = [i for i, q in enumerate(items) if q["split"] == split]
                metric = evaluate(scores[indices], chunks, [items[i] for i in indices])
                print(f"{method} {split}: Hit@3={metric[0]} MRR@3={metric[1]} "
                      f"encode_seconds={encode_seconds:.4f} search_seconds={search_seconds:.4f}")
            for row, q in zip(scores, items):
                for rank, i in enumerate(np.argsort(-row, kind="stable")[:3], 1):
                    c = chunks[i]
                    logs.append({
                        "strategy": strategy, "method": method, "query": q["query"],
                        "split": q["split"], "rank": rank, "score": float(row[i]),
                        "chunk_id": c["id"], "doc": c["doc"], "source": c["source"],
                        "start": c["start"], "end": c["end"], "text": c["text"],
                        "answerable": q["doc"] is not None
                    })
    if logs:
        with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(logs[0]))
            writer.writeheader()
            writer.writerows(logs)
        print(f"\nWrote retrieval_log.csv ({len(logs)} ranked rows).")
    print("\nUnanswerable queries are reported separately; top-ranked text is only a candidate, not proof.")

if __name__ == "__main__":
    main()
