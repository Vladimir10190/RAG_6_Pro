import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from app.rag import build_index

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Укажите OPENAI_API_KEY в .env перед индексацией.", file=sys.stderr)
    raise SystemExit(2)

try:
    client_options = {}
    if os.getenv("OPENAI_BASE_URL"):
        client_options["base_url"] = os.environ["OPENAI_BASE_URL"]
    count = build_index(OpenAI(**client_options))
except Exception as exc:
    print(f"Не удалось создать индекс: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
print(f"Индекс обновлён: {count} смысловых фрагментов.")
