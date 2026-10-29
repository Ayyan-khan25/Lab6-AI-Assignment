# Week 06 — Lab 06: NLP Fundamentals

## Included tasks
- `task1_tokens.py`: whitespace and Unicode word-like tokenization comparison.
- `guided_chatbot.py`: conservative FAQ chatbot with exact answers and candidate suggestions.
- `task2_chunks.py`: fixed windows, overlap, sentence-aware chunks, IDs, and character offsets.
- `task3_retrieval.py`: compares chunking strategies and TF-IDF/embedding retrieval; exports `retrieval_log.csv`.
- `document_chatbot.py`: confirmation-based evidence display.
- `corpus.json`, `queries.json`: synthetic educational documents and labeled queries.
- `lab6_search.py`: shared retrieval helper.
- `requirements.txt`: Python dependencies.

## Setup (Python 3.10+)
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Install packages:
```bash
python -m pip install -r requirements.txt
```

## Run the tasks
```bash
python task1_tokens.py
python guided_chatbot.py
python task1_chatbot_check.py
python task2_chunks.py
python task3_retrieval.py
python document_chatbot.py
```
For semantic FAQ retrieval:
```bash
python guided_chatbot.py --backend embedding
```

The first embedding run downloads `sentence-transformers/all-MiniLM-L6-v2` and needs internet access. CPU is supported; a GPU is not required. If model download is unavailable, Task 3 still runs its TF-IDF comparison and explicitly reports that embeddings were skipped. Do not report missing embedding metrics as completed results.

## Important notes
All policy text is synthetic teaching data, not real university or company policy. Similarity scores are not probabilities or proof of correctness. Review source passages and exceptions before using retrieved evidence. The corpus is deliberately compact for a runnable starter; expand each document to 150–250 words if your instructor requires strict length checks.
