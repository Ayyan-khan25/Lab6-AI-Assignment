"""Task 1 FAQ chatbot: exact answers, otherwise suggestions."""
from lab6_search import Retriever

FAQ = [
    {"id":"faq:password", "question":"How do I reset my password?",
     "answer":"Use the recovery link on the login page.", "source":"it_help.txt"},
    {"id":"faq:submit", "question":"Where do I submit my assignment?",
     "answer":"Use the assignment upload page in the learning portal.", "source":"assessment.txt"},
    {"id":"faq:late", "question":"Can illness justify a late submission?",
     "answer":"Request an extension with medical evidence before the deadline.", "source":"handbook.txt"},
    {"id":"faq:wifi", "question":"How do I connect to campus Wi-Fi?",
     "answer":"Choose the campus network and follow the IT team's account instructions.", "source":"it_help.txt"},
    {"id":"faq:office", "question":"Where can I find office hours?",
     "answer":"Check the course page or ask the instructor for current office hours.", "source":"course_guide.txt"},
]

def build_bot(backend="tfidf"):
    records = [dict(item, text=item["question"] + " " + item["answer"]) for item in FAQ]
    return Retriever(records, backend)

def reply(engine, query):
    if not query.strip():
        return "Please enter a question."
    for item in FAQ:
        if query.strip().casefold() == item["question"].casefold():
            return item["answer"] + " [source: " + item["source"] + "]"
    hits = engine.search(query, k=3)
    if not hits:
        return "No matching FAQ. Please contact the help desk."
    lines = ["Possible matches; choose or rephrase your question:"]
    for item, score in hits:
        lines.append(f'{item["question"]} (similarity {score:.3f})')
    return "\n".join(lines)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tfidf", "embedding"], default="tfidf")
    args = parser.parse_args()
    engine = build_bot(args.backend)
    print("FAQ assistant. Type exit to stop.")
    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip().lower() == "exit":
            break
        print("Bot:", reply(engine, query))
