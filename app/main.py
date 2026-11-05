import os
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from app.rag import Retriever, KB_DIR, build_index

load_dotenv()
app = FastAPI(title="HelpDesk Cloud RAG", version="1.0.0")
update_lock = threading.Lock()
WEB_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class DocumentUpload(BaseModel):
    filename: str = Field(min_length=4, max_length=180)
    content: str = Field(min_length=1, max_length=1_000_000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    no_answer: bool
    diagnostics: dict[str, list[dict[str, Any]]]


@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    try:
        Retriever()
        ready = True
    except Exception:
        ready = False
    return {"status": "ok", "index_ready": ready, "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.get("/api/knowledge")
def knowledge_status():
    documents = [{"filename": p.name, "size": p.stat().st_size} for p in sorted(KB_DIR.glob("*.md")) if p.is_file() and not p.is_symlink()]
    indexing = update_lock.locked()
    try:
        Retriever()
        state = "ready"
        message = "Индекс актуален."
    except Exception:
        state = "stale"
        message = "Индекс отсутствует или устарел. Запустите индексацию."
    return {"documents": documents, "index_state": state, "message": message, "indexing": indexing}


@app.post("/api/knowledge/documents")
def upload_document(request: DocumentUpload):
    name = request.filename
    if (Path(name).name != name or not name.endswith(".md") or name.startswith(".")
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.endswith((' ', '.'))
        or re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', name, re.IGNORECASE)):
        raise HTTPException(422, "Укажите корректное имя файла с расширением .md.")
    if not request.content.strip() or len(request.content.encode("utf-8")) > 1_000_000:
        raise HTTPException(422, "Документ должен быть непустым и не больше 1 МБ в UTF-8.")
    if not update_lock.acquire(blocking=False):
        raise HTTPException(409, "Дождитесь завершения обновления базы знаний.")
    temp = KB_DIR / f".upload-{uuid.uuid4().hex}.tmp"
    try:
        KB_DIR.mkdir(parents=True, exist_ok=True)
        target = KB_DIR / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise HTTPException(422, "Недопустимый файл назначения.")
        replaced = target.exists()
        temp.write_text(request.content, encoding="utf-8")
        os.replace(temp, target)
        return {"filename": name, "replaced": replaced, "message": "Документ сохранён. Запустите индексацию, чтобы использовать изменения."}
    finally:
        temp.unlink(missing_ok=True)
        update_lock.release()


@app.post("/api/knowledge/reindex")
def reindex():
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "Не настроен ключ API для создания embeddings.")
    if not update_lock.acquire(blocking=False):
        raise HTTPException(409, "Обновление базы знаний уже выполняется.")
    try:
        options = {"base_url": os.environ["OPENAI_BASE_URL"]} if os.getenv("OPENAI_BASE_URL") else {}
        count = build_index(OpenAI(**options))
        return {"chunks": count, "message": f"Индексация завершена: {count} фрагментов."}
    except Exception as exc:
        raise HTTPException(503, f"Не удалось построить индекс: {exc}") from exc
    finally:
        update_lock.release()


@app.get("/api/documents/{filename}")
def document(filename: str):
    if Path(filename).name != filename or not filename.endswith(".md"):
        raise HTTPException(404, "Документ не найден")
    path = KB_DIR / filename
    if not path.is_file() or path.is_symlink():
        raise HTTPException(404, "Документ не найден")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(422, "Введите вопрос.")
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "Не настроен ключ OpenAI API. Добавьте OPENAI_API_KEY в файл .env.")
    try:
        retriever = Retriever()
        options = {"base_url": os.environ["OPENAI_BASE_URL"]} if os.getenv("OPENAI_BASE_URL") else {}
        client = OpenAI(**options)
        top_k = max(1, min(20, int(os.getenv("RAG_TOP_K", "4"))))
        diagnostics = retriever.search(client, question, top_k, float(os.getenv("RAG_MIN_SCORE", "0.25")))
        matches = diagnostics["context"]
        fallback = "В базе знаний нет достаточной информации для ответа."
        if not matches:
            return ChatResponse(answer=fallback, sources=[], no_answer=True, diagnostics=diagnostics)
        context = "\n\n".join(f"[S{m['id']}] {m['document']} — {m['section']}\n{m['text']}" for m in matches)
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0,
            messages=[
                {"role": "system", "content": "Ты помощник по документации. Отвечай по-русски исключительно на основе фрагментов. Обязательно указывай существенные условия и исключения, относящиеся к ответу. Не делай предположений. Если ответа нет или данных недостаточно, верни JSON: {\"answer\":\"В базе знаний нет достаточной информации для ответа.\",\"source_ids\":[]}. Иначе верни только JSON с полями answer (ответ) и source_ids (массив идентификаторов вида S12 только тех фрагментов, которые подтверждают ответ). Не используй сведения вне фрагментов."},
                {"role": "user", "content": f"Вопрос: {question}\n\nФрагменты базы знаний:\n{context}"},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        try:
            result = json.loads(raw)
            answer = result["answer"].strip()
            ids = result["source_ids"]
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                raise ValueError("Некорректные ссылки")
            allowed = {f"S{m['id']}": m for m in matches}
            if any(i not in allowed for i in ids):
                raise ValueError("Несуществующая ссылка")
            sources = [allowed[i] for i in dict.fromkeys(ids)]
            no_answer = not sources or not answer or "нет достаточной информации" in answer.casefold()
            if no_answer:
                answer, sources = fallback, []
        except (ValueError, KeyError, TypeError, AttributeError):
            answer, sources, no_answer = fallback, [], True
        return ChatResponse(answer=answer, sources=sources, no_answer=no_answer, diagnostics=diagnostics)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, f"Ошибка обращения к API или индексу: {exc}") from exc
