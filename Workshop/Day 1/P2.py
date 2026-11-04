
import os
import time
import random
import httpx

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors


# ============================================================
# 1. GEMINI API CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-3.8-flash"


# ============================================================
# 2. REUSABLE ASK FUNCTION WITH ERROR HANDLING
# ============================================================

def ask(prompt: str, system: str = "",
        model: str = MODEL, temp: float = 0.7) -> str:

    config = types.GenerateContentConfig(
        temperature=temp,
        system_instruction=system if system else None,
    )

    max_attempts = 4

    for attempt in range(max_attempts):

        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )

            if not resp.text:
                raise ValueError("Gemini returned an empty response")

            return resp.text.strip()

        except (errors.APIError, httpx.TransportError) as e:

            if attempt == max_attempts - 1:
                print(f"[ERROR] API request failed: {e}")
                return "API request failed after retries."

            wait_time = (2 ** (attempt + 1)) + random.uniform(0, 1)

            print(f"[RETRY] Attempt {attempt + 1} failed.")
            print(f"[RETRY] Waiting {wait_time:.1f} seconds...")

            time.sleep(wait_time)


# ============================================================
# TECHNIQUE 1: ZERO-SHOT PROMPTING
# ============================================================

zero_shot_prompt = """
Classify the sentiment of this review as Positive, Negative or Neutral.

Review:
"The course materials were well structured but the lab machines were slow."

Return only one sentiment label.
"""

result = ask(zero_shot_prompt)

print(f"\n[Zero-Shot] {result}")

time.sleep(2)


# ============================================================
# TECHNIQUE 2: FEW-SHOT PROMPTING
# ============================================================

few_shot_prompt = """
Classify the sentiment of a review as Positive, Negative, or Neutral.

Examples:

Review: "Absolutely loved the hands-on labs, learned so much!"
-> Positive

Review: "Confusing explanations and the code didn't run."
-> Negative

Review: "It was okay, nothing special."
-> Neutral

Now classify:

Review: "The course materials were well-structured but the lab machines were slow."

Return only the sentiment label.
"""

result = ask(few_shot_prompt, temp=0.0)

print(f"\n[Few-Shot] {result}")

time.sleep(2)


# ============================================================
# TECHNIQUE 3A: EXPLICIT CHAIN OF THOUGHT
# ============================================================

cot_prompt = """
A student submitted 3 assignments worth 20 marks each
and 2 exams worth 50 marks each.

They scored:
18, 15, 17 on assignments
42, 38 on exams.

Calculate the total percentage.

Show the calculation:
1. Calculate total marks obtained.
2. Calculate maximum possible marks.
3. Compute the percentage.
"""

result = ask(cot_prompt, temp=0.0)

print(f"\n[Explicit CoT]\n{result}")

time.sleep(2)


# ============================================================
# TECHNIQUE 3B: ZERO-SHOT CHAIN OF THOUGHT
# ============================================================

zero_shot_cot_prompt = """
A student submitted 3 assignments worth 20 marks each
and 2 exams worth 50 marks each.

They scored:
18, 15, 17 on assignments
42, 38 on exams.

What is their total percentage?

Explain the calculation step by step.
"""

result = ask(zero_shot_cot_prompt, temp=0.0)

print(f"\n[Zero-Shot CoT]\n{result}")

time.sleep(2)


# ============================================================
# TECHNIQUE 4: ROLE PROMPTING
# ============================================================

plain_prompt = """
Should we use a vector database or a relational database
for our RAG system?
"""

# Without role

plain_result = ask(plain_prompt)

# With role

role_result = ask(
    plain_prompt,
    system=(
        "You are a senior database architect with 15 years "
        "of experience in production AI systems. "
        "Give a direct recommendation with engineering tradeoffs. "
        "Be concise."
    )
)

print(f"\n[No Role]\n{plain_result}")

print(f"\n[With Role]\n{role_result}")

time.sleep(2)


# ============================================================
# TECHNIQUE 5: STRUCTURED OUTPUT CONTRACT
# ============================================================

unstructured_prompt = """
Compare Chroma, FAISS, and Pinecone as vector databases.
"""

unstructured_result = ask(unstructured_prompt)

structured_prompt = """
Compare Chroma, FAISS, and Pinecone as vector databases.

Return your answer as a Markdown table with exactly
these columns:

| Database | Type | Best For | Limitation |

Do not add any text before or after the table.
"""

structured_result = ask(structured_prompt, temp=0.0)

print(f"\n[No Format]\n{unstructured_result[:300]}...")

print(f"\n[With Format Contract]\n{structured_result}")

time.sleep(2)


# ============================================================
# TECHNIQUE 6: SELF-CONSISTENCY
# ============================================================

reasoning_prompt = """
A RAG pipeline retrieves 5 chunks for a question.

Chunk relevance scores (cosine similarity):
[0.91, 0.88, 0.72, 0.65, 0.41].

Should we use the top-3 or top-5 chunks?

Give exactly one answer:
top-3 or top-5.
"""

votes: list[str] = []

for i in range(5):

    answer = ask(reasoning_prompt, temp=0.7)

    if answer == "API request failed after retries.":
        print(f"[Vote {i+1}] Failed")
        continue

    answer_lower = answer.lower()

    if "top-3" in answer_lower:
        normalised = "top-3"

    elif "top-5" in answer_lower:
        normalised = "top-5"

    else:
        print(f"[Vote {i+1}] Invalid response: {answer}")
        continue

    votes.append(normalised)

    print(f"[Vote {i+1}] {normalised}")

    time.sleep(2)


if votes:

    majority = max(set(votes), key=votes.count)

    print(f"\n[Self-Consistency] Votes: {votes}")
    print(f"Majority: {majority}")

else:

    print("\n[Self-Consistency] No valid votes received.")


time.sleep(2)


# ============================================================
# TECHNIQUE 7: SYSTEM INSTRUCTION VS USER MESSAGE
# ============================================================

SYSTEM = (
    "You are a technical documentation assistant for an engineering course. "
    "Rules: "
    "(1) Always answer in under 3 sentences. "
    "(2) Use precise technical vocabulary. "
    "(3) If the question is outside your domain, say 'Out of scope.'"
)

questions = [
    "What is HNSW and why does Chroma use it?",
    "What is the best recipe for biryani?",
    "What does MMR do in a retrieval pipeline?"
]

for q in questions:

    answer = ask(q, system=SYSTEM)

    print(f"\nQ: {q}")
    print(f"A: {answer}")

    time.sleep(2)


# ============================================================
# TECHNIQUE 8: CONTEXT WINDOW BUDGETING
# ============================================================

retrieved_chunks = [

    "Chroma uses HNSW (Hierarchical Navigable Small World) "
    "graphs for approximate nearest-neighbour search.",

    "FAISS is a library developed by Meta AI for efficient "
    "similarity search and clustering of dense vectors. " * 8,

    "Pinecone is a managed vector database that handles "
    "infrastructure, scaling, and replication automatically.",

    "MMR (Maximal Marginal Relevance) balances relevance "
    "and diversity when selecting retrieval results.",

    "Cosine similarity measures the angle between two vectors, "
    "not their magnitude. Values range from -1 to 1." * 5,
]


MAX_CONTEXT_CHARS = 800

QUESTION = "What is HNSW and how does it relate to Chroma?"


# Greedy context packing

packed_chunks = []

total = 0

for chunk in retrieved_chunks:

    formatted_chunk = f"[{len(packed_chunks)+1}] {chunk}"

    separator_length = 2 if packed_chunks else 0

    available = MAX_CONTEXT_CHARS - total - separator_length

    if available <= 0:
        break

    if len(formatted_chunk) <= available:

        packed_chunks.append(formatted_chunk)

        total += separator_length + len(formatted_chunk)

    else:

        if available > 100:

            packed_chunks.append(formatted_chunk[:available])

            total += separator_length + available

        break


context_block = "\n\n".join(packed_chunks)


budgeted_prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.

If the answer is not in the context, say "I don't know."

Context:

{context_block}

Question: {QUESTION}

Answer:
"""

answer = ask(budgeted_prompt, temp=0.0)

print("\n[Budgeted RAG Prompt]")

print(f"Context characters used: {total}")

print(f"Answer: {answer}")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n==========================================")
print(" ALL PROMPT ENGINEERING TECHNIQUES DONE ")
print("==========================================")
