"""Live RAG evaluation: python tests/evaluate_live.py (uses configured API)."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import ChatRequest, chat

QUESTIONS = [
    'Как восстановить забытый пароль?',
    'Сколько времени отводится на критическую заявку?',
    'Может ли оператор экспортировать данные?',
    'Что делать при ошибке HTTP 429?',
    'Какой максимальный размер вложения?',
    'Как подключить корпоративную почту?',
    'Поддерживает ли сервис уведомления в Telegram?',
    'Сколько стоит тариф Team?',
    'Не помню данные для входа: как заново установить пароль?',
    'Поддерживает ли сервис уведомления в WhatsApp?',
]
def run(question):
    try:
        return {'question': question, **chat(ChatRequest(question=question)).model_dump()}
    except Exception as exc:
        return {'question': question, 'error': str(exc)}

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, QUESTIONS))
    path = Path(__file__).parent / 'live_results.json'
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    for item in results:
        print(json.dumps({k: item.get(k) for k in ('question', 'answer', 'no_answer', 'error')}, ensure_ascii=True))
    print('Full diagnostics:', path)
