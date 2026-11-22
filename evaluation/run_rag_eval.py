import json
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from config.settings import RAG_QUERIES_PATH, RAW_RUN_PATH, RESULTS_CSV_PATH

load_dotenv()

from ..src.medmate.rag.pipeline import run_rag_query

judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_pipeline_on_golden_set(golden):
    records = []
    out_f = open(RAW_RUN_PATH, "w", encoding="utf-8")
    min_interval = 60.0 / 10  # 6 seconds between calls -> max 10/min

    try:
        for i, item in enumerate(golden):
            start = time.time()
            question = item["question"]
            try:
                answer, contexts, sources = run_rag_query(question)
            except Exception as e:
                print(f"[{i}] Pipeline error on '{question[:50]}...': {e}")
                elapsed = time.time() - start
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                continue

            record = {
                "id": item["id"],
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "sources": sources,
                "ground_truth": item["expected_answer"],
                "category": item.get("category"),
                "difficulty": item.get("difficulty"),
            }
            records.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if i % 10 == 0:
                print(f"Ran {i}/{len(golden)} queries through pipeline")

            elapsed = time.time() - start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
    finally:
        out_f.close()
    return records

def main():
    golden = load_jsonl(RAG_QUERIES_PATH)
    print(f"Loaded {len(golden)} golden queries")

    records = run_pipeline_on_golden_set(golden)
    print(f"Pipeline completed for {len(records)}/{len(golden)} queries")

    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in records
    ])

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
    )

    df = result.to_pandas()
    for i, r in enumerate(records):
        df.loc[i, "id"] = r["id"]
        df.loc[i, "category"] = r["category"]
        df.loc[i, "difficulty"] = r["difficulty"]

    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nSaved results to {RESULTS_CSV_PATH}")
    print(result)


if __name__ == "__main__":
    main()