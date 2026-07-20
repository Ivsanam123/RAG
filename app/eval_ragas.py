import asyncio
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
JUDGE_MODEL = "gpt-4o-mini"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    decoder = json.JSONDecoder(strict=False)
    results = []
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx] in " \t\n\r":
            idx += 1
        if idx >= length:
            break
        obj, end = decoder.raw_decode(text, idx)
        results.append(obj)
        idx = end
    return results


def judge_faithfulness(answer: str, contexts: list[str]) -> dict:
    context_block = "\n\n---\n\n".join(contexts)
    prompt = f"""You are grading whether an AI-generated answer is faithful to the given context (i.e. not hallucinating facts not present in the context).

Context:
{context_block}

Answer:
{answer}

Score faithfulness from 1-5:
1 = answer contains claims completely unsupported by context
5 = every claim in the answer is directly supported by the context

Respond ONLY with JSON: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def judge_relevancy(question: str, answer: str) -> dict:
    prompt = f"""Grade how relevant this answer is to the question asked.

Question: {question}
Answer: {answer}

Score relevancy from 1-5:
1 = answer doesn't address the question at all
5 = answer directly and fully addresses the question

Respond ONLY with JSON: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def judge_correctness(answer: str, reference: str) -> dict:
    prompt = f"""Compare this generated answer against a reference (ground-truth) answer.

Reference answer: {reference}
Generated answer: {answer}

Score correctness from 1-5:
1 = generated answer contradicts or misses the key facts in the reference
5 = generated answer matches the key facts in the reference

Respond ONLY with JSON: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def evaluate_rag_system(test_path="../seed/qna_test.json"):
    test_data = load_jsonl(test_path)
    results = []

    for i, item in enumerate(test_data, start=1):
        question = item["question"]
        reference_answer = item["answer"]

        res = requests.post("http://localhost:8000/ask", json={"question": question}).json()
        answer = res["answer"]
        contexts = res["contexts"]

        faithfulness = judge_faithfulness(answer, contexts)
        relevancy = judge_relevancy(question, answer)
        correctness = judge_correctness(answer, reference_answer)

        row = {
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": answer,
            "retrieved_contexts": contexts,
            "faithfulness": faithfulness["score"],
            "relevancy": relevancy["score"],
            "correctness": correctness["score"],
            "faithfulness_reason": faithfulness["reason"],
            "relevancy_reason": relevancy["reason"],
            "correctness_reason": correctness["reason"],
        }
        results.append(row)

        print(f"[{i}/{len(test_data)}] {question[:60]}")
        print(f"   faithfulness={row['faithfulness']}  relevancy={row['relevancy']}  correctness={row['correctness']}")

    print("\n=== Manual Eval Results ===")
    print(f"{'#':<4}{'Faith':<8}{'Relev':<8}{'Correct':<8}Question")
    for i, r in enumerate(results, start=1):
        print(f"{i:<4}{r['faithfulness']:<8}{r['relevancy']:<8}{r['correctness']:<8}{r['question'][:50]}")

    n = len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_relev = sum(r["relevancy"] for r in results) / n
    avg_correct = sum(r["correctness"] for r in results) / n

    print("\n📈 Averages:")
    print(f"- faithfulness: {avg_faith:.2f}")
    print(f"- relevancy: {avg_relev:.2f}")
    print(f"- correctness: {avg_correct:.2f}")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved detailed results to eval_results.json")


if __name__ == "__main__":
    evaluate_rag_system()