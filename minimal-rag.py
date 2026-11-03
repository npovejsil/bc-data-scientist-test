"""
This script implements a minimal Retrieval-Augmented Generation (RAG) pipeline. It retrieves relevant documents
from a predefined knowledge base using TF-IDF vectorization and cosine similarity, then generates a contextual
response to a user-provided question using a GPT-2 language model. The script evaluates the quality of the
response based on relevance, references, and length, providing a critique and confidence score. It can be run
from the command line with options to specify the number of documents to retrieve and output format.
"""

# minimal rag

import argparse
import sys
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

NO_DOCS_ANSWER = "No document matches found, cannot answer."
NO_MODEL_ANSWER = "The model returned no answer."

docs = [
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

vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform([x["text"] for x in docs])

retrieved_docs = []
answer = ""
problems = []
score = 1.0
is_valid = False
temp_storage = {}

parser = argparse.ArgumentParser(
    description="Answer a question from a small knowledge base and critique the answer.",
)
parser.add_argument("question", help="the question to answer")
parser.add_argument(
    "-k",
    type=int,
    default=3,
    help="maximum number of documents to retrieve (default: 3)",
)
parser.add_argument(
    "--json",
    dest="output_json",
    action="store_true",
    help="print the result as JSON instead of a readable report",
)
args = parser.parse_args()
if args.k < 1:
    parser.error(f"-k must be at least 1, got {args.k}")

question = args.question
k = args.k
use_json = args.output_json

retrieved_docs = []
answer = ""
problems = []
score = 1.0
temp_storage["question"] = question


def get_data():
    global retrieved_docs, temp_storage
    query_vec = vectorizer.transform([question])
    similarities = cosine_similarity(query_vec, vectors)[0]
    indices = np.argsort(similarities)[-k:][::-1]
    temp_storage["sims"] = similarities

    for i in indices:
        if similarities[i] > 0:
            retrieved_docs.append(
                {
                    "citation_number": len(retrieved_docs) + 1,
                    "doc": docs[i],
                    "score": float(similarities[i]),
                }
            )
            if len(retrieved_docs) >= k:
                break


get_data()

if len(retrieved_docs) == 0:
    answer = NO_DOCS_ANSWER
    problems.append("no docs")
    score = score * 0.4
    is_valid = False
    if use_json:
        print(
            json.dumps(
                {
                    "answer": answer,
                    "citations": retrieved_docs,
                    "critique": {
                        "passed": is_valid,
                        "confidence": score,
                        "problems": problems,
                    },
                },
                indent=2,
            )
        )
    else:
        print("\nQ: {}".format(question))
        print(f"\nA: {answer}\n")
        print("Citations: None")
        print("\nCritique: ISSUES (confidence: {:.2f})".format(score))
        print("Problems: %s" % ", ".join(problems))
    sys.exit(0)


def call_model():
    global answer, temp_storage
    ctx = ""
    counter = 0
    for r in retrieved_docs:
        counter += 1
        ctx += f"[{counter}] {r['doc']['text']}\n\n"

    temp_storage["context"] = ctx
    prompt = "Context:\n{}\nQuestion: {}\n\nAnswer the question based only on the context above. Cite sources using [1], [2], etc. Be concise.\n\nAnswer:".format(
        ctx, temp_storage["question"]
    )
    print("loading model...", file=sys.stderr)
    try:
        llm = pipeline("text-generation", model="gpt2", max_new_tokens=256)
    except Exception as error:
        sys.exit(f"Could not load model 'gpt2': {error}")
    print("model loaded", file=sys.stderr)

    output = llm(
        prompt,
        return_full_text=False,
        do_sample=False,
        no_repeat_ngram_size=3,
        pad_token_id=llm.tokenizer.eos_token_id,
    )
    answer = output[0]["generated_text"].strip()

    if answer == "":
        answer = NO_MODEL_ANSWER


call_model()


def validate_output():
    global problems, score, is_valid
    problems = []
    score = 1.0

    if answer == NO_MODEL_ANSWER:
        problems.append("no answer")
        score = score * 0.5

    if len(answer) < 20:
        problems.append("short answer")
        score = score * 0.3

    refs = [
        number
        for number in map(int, re.findall(r"\[(\d+)\]", answer))
        if 1 <= number <= len(retrieved_docs)
    ]

    if len(retrieved_docs) > 0:
        if len(refs) == 0:
            problems.append("no citations")
            score = score * 0.7

        avg = 0
        for r in retrieved_docs:
            avg += r["score"]
        avg = avg / len(retrieved_docs)

        if avg < 0.1:
            problems.append("low relevance")
            score = score * 0.6

    if score > 0.6:
        if len(problems) < 3:
            is_valid = True
        else:
            is_valid = False
    else:
        is_valid = False


validate_output()

result = {
    "answer": answer,
    "citations": retrieved_docs,
    "critique": {"passed": is_valid, "confidence": score, "problems": problems},
}

if use_json:
    print(json.dumps(result, indent=2))
else:
    print("\nQ: {}".format(question))
    print(f"\nA: {answer}\n")
    print("Citations:")

    for r in retrieved_docs:
        doc_text = r["doc"]["text"]
        if len(doc_text) > 80:
            doc_text = doc_text[:80] + "..."
        print(
            "  [{}] (doc {}, score: {:.3f}) {}".format(
                r["citation_number"], r["doc"]["id"], r["score"], doc_text
            )
        )

    if is_valid:
        print("\nCritique: OK (confidence: {:.2f})".format(score))
    else:
        print(f"\nCritique: ISSUES (confidence: {score:.2f})")

    if len(problems) > 0:
        print("Problems: %s" % ", ".join(problems))
