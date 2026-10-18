
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = "gemini-3.8-flash"


def generate_text(prompt: str) -> str:
    """Send a prompt to Gemini and return its text response."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text
