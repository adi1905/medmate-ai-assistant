import json
import time
import random
from qdrant_client import QdrantClient
from openai import OpenAI
from config.settings import QDRANT_PATH
from dotenv import load_dotenv

load_dotenv()

# ---- Config ----
COLLECTION_NAME = "medmate_rag_docs"
TEXT_FIELD = "content"
OUTPUT_PATH = r"C:\Users\adity\OneDrive\Documents\MedMate Clinic Copilot\src\medmate\evaluation\golden_dataset\rag_queries.jsonl"

SAMPLE_SIZE = 150
MULTI_HOP_PAIRS = 30
MIN_CHARS = 200
MAX_CHARS = 3000
QUESTIONS_PER_CHUNK = 1

client = QdrantClient(path=QDRANT_PATH)
openai_client = OpenAI()

# ---- Open output file once, append as we go ----
out_f = open(OUTPUT_PATH, "w", encoding="utf-8")

def write_record(record):
    out_f.write(json.dumps(record) + "\n")
    out_f.flush()  # ensure it's on disk immediately, even if script crashes later

# ---- 1. Pull points from Qdrant ----
def fetch_all_points(collection_name, batch_size=100):
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points

all_points = fetch_all_points(COLLECTION_NAME)
print(f"Fetched {len(all_points)} points")

# ---- 2. Filter by chunk length BEFORE sampling ----
def valid_chunk(point):
    text = point.payload.get(TEXT_FIELD, "") or ""
    return MIN_CHARS <= len(text) <= MAX_CHARS

eligible_points = [p for p in all_points if valid_chunk(p)]
print(f"{len(eligible_points)}/{len(all_points)} points passed length filter "
      f"({MIN_CHARS}-{MAX_CHARS} chars)")

random.seed(42)
sampled_points = random.sample(eligible_points, min(SAMPLE_SIZE, len(eligible_points)))

# ---- 3a. Single-chunk prompt ----
SINGLE_HOP_SYSTEM_PROMPT = """You are creating evaluation data for a RAG system.
Given a passage of text, generate one realistic user question that:
- Can be answered using ONLY the information in the passage
- Sounds like something a real user would ask (natural phrasing, not "According to the text...")
- Has a clear, factual answer grounded in the passage
- Is NOT answerable with just a single word or yes/no — it should require a real sentence answer

If the passage is too generic, too short, or lacks any concrete fact (e.g. it's just a heading,
disclaimer, or table of contents), respond with {"skip": true} instead.

Respond ONLY in this exact JSON format, no markdown, no extra text:
{"question": "...", "expected_answer": "..."}
"""

def generate_single_hop(chunk_text):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SINGLE_HOP_SYSTEM_PROMPT},
                {"role": "user", "content": f"Passage:\n{chunk_text}"}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        if result.get("skip"):
            return None
        return result
    except Exception as e:
        print(f"Error generating single-hop question: {e}")
        return None

# ---- 3b. Multi-hop prompt (2 chunks) ----
MULTI_HOP_SYSTEM_PROMPT = """You are creating evaluation data for a RAG system.
You are given TWO passages from the same knowledge base. Generate one realistic user
question that REQUIRES information from BOTH passages to answer correctly — a question
answerable from only one passage is not acceptable.

If the two passages are unrelated and no meaningful combined question is possible,
respond with {"skip": true}.

Respond ONLY in this exact JSON format, no markdown, no extra text:
{"question": "...", "expected_answer": "..."}
"""

def generate_multi_hop(chunk_a, chunk_b):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MULTI_HOP_SYSTEM_PROMPT},
                {"role": "user", "content": f"Passage A:\n{chunk_a}\n\nPassage B:\n{chunk_b}"}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        if result.get("skip"):
            return None
        return result
    except Exception as e:
        print(f"Error generating multi-hop question: {e}")
        return None

def classify_difficulty(chunk_text, is_multi_hop=False):
    if is_multi_hop:
        return "multi_hop"
    return "simple" if len(chunk_text.split()) < 80 else "reasoning"

single_hop_count = 0
multi_hop_count = 0

try:
    # ---- 4a. Single-hop generation ----
    for i, point in enumerate(sampled_points):
        payload = point.payload
        chunk_text = payload.get(TEXT_FIELD, "")

        for _ in range(QUESTIONS_PER_CHUNK):
            result = generate_single_hop(chunk_text)
            if not result or "question" not in result or "expected_answer" not in result:
                continue

            record = {
                "id": f"rag_{i:04d}",
                "question": result["question"],
                "expected_answer": result["expected_answer"],
                "expected_contexts": [chunk_text],
                "category": payload.get("category", "general"),
                "difficulty": classify_difficulty(chunk_text),
                "source_point_ids": [str(point.id)],
            }
            write_record(record)
            single_hop_count += 1

        if i % 20 == 0:
            print(f"Single-hop processed {i}/{len(sampled_points)} (written: {single_hop_count})")
        time.sleep(0.1)

    # ---- 4b. Multi-hop generation ----
    multi_hop_candidates = random.sample(
        eligible_points, min(MULTI_HOP_PAIRS * 2, len(eligible_points))
    )
    pairs = [
        (multi_hop_candidates[j], multi_hop_candidates[j + 1])
        for j in range(0, len(multi_hop_candidates) - 1, 2)
    ][:MULTI_HOP_PAIRS]

    for k, (point_a, point_b) in enumerate(pairs):
        text_a = point_a.payload.get(TEXT_FIELD, "")
        text_b = point_b.payload.get(TEXT_FIELD, "")

        result = generate_multi_hop(text_a, text_b)
        if not result or "question" not in result or "expected_answer" not in result:
            continue

        record = {
            "id": f"rag_mh_{k:04d}",
            "question": result["question"],
            "expected_answer": result["expected_answer"],
            "expected_contexts": [text_a, text_b],
            "category": point_a.payload.get("category", "general"),
            "difficulty": "multi_hop",
            "source_point_ids": [str(point_a.id), str(point_b.id)],
        }
        write_record(record)
        multi_hop_count += 1

        if k % 10 == 0:
            print(f"Multi-hop processed {k}/{len(pairs)} (written: {multi_hop_count})")
        time.sleep(0.1)

finally:
    out_f.close()
    client.close()
    print(f"\nDone. Total written: {single_hop_count + multi_hop_count}")
    print(f"  Single-hop: {single_hop_count}")
    print(f"  Multi-hop:  {multi_hop_count}")