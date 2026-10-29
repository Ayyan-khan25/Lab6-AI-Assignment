"""Task 1: compare whitespace and Unicode word-like tokenization."""
import re

SAMPLES = [
    "AI, search!",
    "I can't log in.",
    "help@example.com",
    "CS4944",
    "",
    "یہ ایک مثال ہے",
    "Version 2.1 is ready!",
]

def compare_tokens(text):
    whitespace = text.split()
    word_like = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    print("Original:", repr(text))
    print("Whitespace:", whitespace, "count =", len(whitespace))
    print("Word-like:", word_like, "count =", len(word_like))
    print("-" * 60)

if __name__ == "__main__":
    for sample in SAMPLES:
        compare_tokens(sample)
    print("Optional model-tokenizer comparison for two English examples:")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        for sample in SAMPLES[:2]:
            print(repr(sample), tokenizer.tokenize(sample))
    except Exception as exc:
        print("Model tokenizer comparison skipped:", exc)
        print("Install sentence-transformers and ensure model access to run model-token comparison.")
