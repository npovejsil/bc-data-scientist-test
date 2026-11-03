"""
A minimal Retrieval-Augmented Generation (RAG) pipeline.

1. Retrieve: rank the documents in KNOWLEDGE_BASE against the question with TF-IDF and cosine similarity.
2. Generate: ask a Qwen2.5 Instruct model to answer using only the retrieved documents, citing them as [1], [2], ...
3. Critique: score the answer's confidence based on its length, citations, and the retrieval relevance.

Usage:
    python minimal-rag.py "What is RAG?"              # readable report
    python minimal-rag.py "What is RAG?" -k 2 --json  # JSON only, at most 2 documents
"""

import argparse
import json
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Ungated Hugging Face chat models can work without login here.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256
# Maximum number of documents passed to the LLM.
DEFAULT_K = 3

# Hand-picked heuristics, not tuned. Penalty nearer 1.0 = milder problem.
# Pass requires confidence > 0.6: lone "no citations" passes, lone "low relevance" fails.
MIN_ANSWER_CHARS = 20
MIN_AVG_RELEVANCE = 0.1
MIN_PASSING_CONFIDENCE = 0.6
PROBLEM_PENALTIES = {
    "no docs": 0.4,
    "no answer": 0.5,
    "short answer": 0.3,
    "no citations": 0.7,
    "low relevance": 0.6,
}

NO_DOCS_ANSWER = "No document matches found, cannot answer."
NO_MODEL_ANSWER = "The model returned no answer."

KNOWLEDGE_BASE = [
    {
        "id": "1",
        "text": "Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant whitespace.",
    },
    {
        "id": "2",
        "text": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It is based on standard Python type hints and provides automatic API documentation. FastAPI is one of the fastest Python frameworks available.",
    },
    {
        "id": "3",
        "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It uses algorithms to parse data, learn from it, and make predictions or decisions without being explicitly programmed.",
    },
    {
        "id": "4",
        "text": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate informed, contextual answers. This approach improves accuracy and reduces hallucinations.",
    },
    {
        "id": "5",
        "text": "Vector databases store data as high-dimensional vectors and enable efficient similarity search. They are essential for modern AI applications, particularly in semantic search and retrieval systems. Common examples include Pinecone, Weaviate, and Chroma.",
    },
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Answer a question from a small knowledge base and critique the answer.",
    )
    parser.add_argument("question", help="the question to answer")  # required (no dash)
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help=f"maximum number of documents to retrieve (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",  # default false if no --json flag set
        help="print the result as JSON instead of a readable report",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error(f"-k must be at least 1, got {args.k}")
    return args


def retrieve_docs(question, k):
    """Return up to k matching documents, best first, each as {"citation_number": rank, "doc": ..., "score": similarity}."""
    # Drop stop words so "what"/"is" don't match.
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform([doc["text"] for doc in KNOWLEDGE_BASE])
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, doc_vectors)[0]
    ranked_indices = np.argsort(scores)[::-1]

    retrieved_docs = []
    for i in ranked_indices[:k]:
        assert scores[i] >= 0
        # 0 = no shared words.
        if scores[i] > 0:
            retrieved_docs.append(
                {
                    "citation_number": len(retrieved_docs) + 1,
                    "doc": KNOWLEDGE_BASE[i],
                    "score": float(scores[i]),
                }
            )
    return retrieved_docs


def generate_answer(question, retrieved_docs):
    """Ask the model to answer the question using only the retrieved documents."""
    context = ""
    for retrieved in retrieved_docs:
        context += f"[{retrieved['citation_number']}] {retrieved['doc']['text']}\n\n"

    messages = [
        {
            "role": "system",
            "content": "Answer the question based only on the context provided. End every sentence with the number in brackets of the passage it came from, e.g. [1]. Be concise.",
        },
        # Small models tend to skip citations unless shown one.
        {
            "role": "user",
            "content": "Context:\n[1] Oslo is the capital of Norway.\n\n[2] Paris is the capital of France.\n\nQuestion: What is the capital of France?",
        },
        {"role": "assistant", "content": "Paris is the capital of France [2]."},
        {"role": "user", "content": f"Context:\n{context}\nQuestion: {question}"},
    ]

    # stderr keeps stdout clean for --json.
    print("loading model...", file=sys.stderr)
    try:
        # auto keeps the checkpoint's bfloat16: half the memory of float32.
        llm = pipeline(
            "text-generation",
            model=MODEL_NAME,
            max_new_tokens=MAX_NEW_TOKENS,
            dtype="auto",
        )
    except Exception as error:
        sys.exit(f"Could not load model '{MODEL_NAME}': {error}")
    print("model loaded", file=sys.stderr)

    # Greedy decoding for repeatable answers (most likely token next).
    output = llm(messages, return_full_text=False, do_sample=False)
    return output[0]["generated_text"].strip() or NO_MODEL_ANSWER


def critique_answer(answer, retrieved_docs):
    """Run heuristic quality checks on the answer and score its confidence."""
    problems = []
    if not retrieved_docs:
        # Other checks would be irrelevant if this problem occurs.
        problems.append("no docs")
    else:
        if answer == NO_MODEL_ANSWER:
            problems.append("no answer")

        if len(answer) < MIN_ANSWER_CHARS:
            problems.append("short answer")

        # Only count citations of retrieved docs (use [int] regex expression).
        cited_numbers = [int(number) for number in re.findall(r"\[(\d+)\]", answer)]
        valid_citations = [
            number for number in cited_numbers if 1 <= number <= len(retrieved_docs)
        ]
        if not valid_citations:
            problems.append("no citations")

        average_score = np.mean([retrieved["score"] for retrieved in retrieved_docs])
        if average_score < MIN_AVG_RELEVANCE:
            problems.append("low relevance")

    confidence = 1.0
    for problem in problems:
        confidence *= PROBLEM_PENALTIES[problem]

    return {
        "passed": confidence > MIN_PASSING_CONFIDENCE,
        "confidence": confidence,
        "problems": problems,
    }


def main():
    args = get_args()
    retrieved_docs = retrieve_docs(args.question, args.k)

    if retrieved_docs:
        answer = generate_answer(args.question, retrieved_docs)
    else:
        answer = NO_DOCS_ANSWER
    critique = critique_answer(answer, retrieved_docs)

    if args.output_json:
        result = {"answer": answer, "citations": retrieved_docs, "critique": critique}
        print(json.dumps(result, indent=2))
        return

    print(f"\nQ: {args.question}")
    print(f"\nA: {answer}\n")

    print("Citations:" if retrieved_docs else "Citations: None")
    for retrieved in retrieved_docs:
        doc = retrieved["doc"]
        print(
            f"  [{retrieved['citation_number']}] (doc {doc['id']}, score: {retrieved['score']:.3f}) {doc['text'][:80]}..."
        )

    status = "OK" if critique["passed"] else "ISSUES"
    print(f"\nCritique: {status} (confidence: {critique['confidence']:.2f})")
    if critique["problems"]:
        print(f"Problems: {', '.join(critique['problems'])}")


if __name__ == "__main__":
    main()
