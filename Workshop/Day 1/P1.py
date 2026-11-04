import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found")

client = genai.Client(api_key=GEMINI_API_KEY)

try:
    ping = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Reply with exactly three words: API key works."
    )

    print(f"[PING] {ping.text.strip()}")

except Exception as e:
    print(f"[ERROR] {e}")