"""Document ingestion and semantic retrieval backed by OpenAI embeddings and FAISS."""
from __future__ import annotations

import json
import os
import re
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "БЗ"
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"
MAX_CHARS = 2400


def parse_documents() -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
        heading = path.stem
        section = ""
        body: list[str] = []

        def flush() -> None:
            content = "\n".join(body).strip()
            if not content:
                return
            # Keep table rows and their nearby prose together; split only at paragraph boundaries.
            if len(content) <= MAX_CHARS:
                chunks.append({"text": content, "document": heading, "section": section or heading, "file": path.name})
                return
            paragraphs = re.split(r"\n\s*\n", content)
            part: list[str] = []
            size = 0
            for paragraph in paragraphs:
                if part and size + len(paragraph) > MAX_CHARS:
                    chunks.append({"text": "\n\n".join(part), "document": heading, "section": section or heading, "file": path.name})
                    part, size = [], 0
                part.append(paragraph)
                size += len(paragraph)
            if part:
                chunks.append({"text": "\n\n".join(part), "document": heading, "section": section or heading, "file": path.name})

        for line in text.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if match:
                flush()
                body = []
                if len(match.group(1)) == 1:
                    heading = match.group(2)
                    section = ""
                else:
                    section = match.group(2)
            else:
                body.append(line)
        flush()
    return chunks


def build_index(client: OpenAI) -> int:
    chunks = parse_documents()
    if not chunks:
        raise RuntimeError(f"В папке {KB_DIR} не найдено Markdown-документов")
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), 100):
        batch = chunks[start : start + 100]
        result = client.embeddings.create(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            input=[f"Документ: {item['document']}\nРаздел: {item['section']}\n{item['text']}" for item in batch],
        )
        vectors.extend(item.embedding for item in result.data)
    matrix = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Use Python's wide-character file APIs: FAISS's Windows filename overload
    # can fail when an ancestor directory contains non-ASCII characters.
    # Publish the manifest last: readers see either the old complete index or the new one.
    generation = uuid.uuid4().hex
    new_index = DATA_DIR / f"index-{generation}.faiss"
    manifest_tmp = DATA_DIR / f"chunks-{generation}.tmp"
    try:
        with new_index.open("wb") as index_file:
            faiss.write_index(index, faiss.PyCallbackIOWriter(index_file.write))
        manifest_tmp.write_text(json.dumps({"model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), "chunks": chunks, "index_file": new_index.name}, ensure_ascii=False, indent=2), encoding="utf-8")
        if parse_documents() != chunks:
            raise RuntimeError("Документы изменились во время индексации. Запустите её ещё раз.")
        os.replace(manifest_tmp, CHUNKS_PATH)
    except Exception:
        new_index.unlink(missing_ok=True)
        raise
    finally:
        manifest_tmp.unlink(missing_ok=True)
    return len(chunks)


def tokenize(text: str) -> list[str]:
    # Keep error identifiers and API paths searchable as well as their components.
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


class Retriever:
    def __init__(self) -> None:
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError("Индекс базы знаний не найден. Запустите индексацию.")
        saved = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        index_path = INDEX_PATH
        if isinstance(saved, dict) and saved.get("index_file"):
            name = saved["index_file"]
            if not re.fullmatch(r"index-[0-9a-f]{32}\.faiss", name):
                raise ValueError("Некорректное имя файла индекса")
            index_path = DATA_DIR / name
        with index_path.open("rb") as index_file:
            self.index = faiss.read_index(faiss.PyCallbackIOReader(index_file.read))
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        if isinstance(saved, dict):
            if saved.get("model") != model:
                raise ValueError("Модель embeddings изменилась. Выполните: python -m app.index")
            self.chunks = saved["chunks"]
        else:
            # Existing indices predate source filenames; only accept an exact match.
            current = parse_documents()
            if [{k: v for k, v in c.items() if k != "file"} for c in current] != saved:
                raise ValueError("Документы изменились. Выполните: python -m app.index")
            self.chunks = current
        if self.chunks != parse_documents() or self.index.ntotal != len(self.chunks):
            raise ValueError("Документы или индекс изменились. Выполните: python -m app.index")
        self.tokens = [tokenize(f"{c['document']} {c['section']} {c['text']}") for c in self.chunks]
        self.counts = [Counter(tokens) for tokens in self.tokens]
        self.avg_length = sum(map(len, self.tokens)) / len(self.tokens)
        self.df = Counter(token for counts in self.counts for token in counts)

    def search(self, client: OpenAI, question: str, top_k: int, min_score: float) -> dict[str, list[dict[str, Any]]]:
        # Expand common SLA phrasing so inflected Russian queries also match the table.
        query = question
        if "критическ" in question.casefold() or "sla" in question.casefold():
            query += " Приоритеты SLA критический приоритет целевое время первого ответа решение рабочие часы"
        result = client.embeddings.create(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), input=[query])
        vector = np.asarray([result.data[0].embedding], dtype="float32")
        if vector.shape[1] != self.index.d:
            raise ValueError("Размер embeddings не совпадает с индексом. Выполните: python -m app.index")
        faiss.normalize_L2(vector)
        candidate_k = min(max(top_k * 3, 10), len(self.chunks))
        scores, indices = self.index.search(vector, candidate_k)
        vector_results = [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx >= 0 and float(score) >= min_score]
        query_tokens = set(tokenize(query))
        lexical = []
        for idx, counts in enumerate(self.counts):
            score = 0.0
            for token in query_tokens:
                freq = counts[token]
                if freq:
                    idf = math.log(1 + (len(self.chunks) - self.df[token] + 0.5) / (self.df[token] + 0.5))
                    score += idf * freq * 2.5 / (freq + 1.5 * (0.25 + 0.75 * len(self.tokens[idx]) / self.avg_length))
            if score > 0:
                lexical.append((idx, score))
        lexical.sort(key=lambda pair: (-pair[1], pair[0]))
        lexical = lexical[:candidate_k]
        fused: dict[int, float] = {}
        for results in (vector_results, lexical):
            for rank, (idx, _) in enumerate(results, 1):
                fused[idx] = fused.get(idx, 0.0) + 1 / (60 + rank)
        final = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
        def decorate(results):
            return [{**self.chunks[idx], "id": idx, "rank": rank, "score": score} for rank, (idx, score) in enumerate(results, 1)]
        return {"vector": decorate(vector_results), "bm25": decorate(lexical), "context": decorate(final)}
