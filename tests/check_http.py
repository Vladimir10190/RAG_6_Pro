import json, urllib.request, concurrent.futures
from pathlib import Path
questions=['Как восстановить забытый пароль?', 'Сколько времени отводится на критическую заявку?', 'Может ли оператор экспортировать данные?', 'Что делать при ошибке HTTP 429?', 'Какой максимальный размер вложения?', 'Как подключить корпоративную почту?', 'Поддерживает ли сервис уведомления в Telegram?', 'Сколько стоит тариф Team?']
def ask(q):
 try:
  request=urllib.request.Request('http://127.0.0.1:8000/api/chat',data=json.dumps({'question':q},ensure_ascii=False).encode('utf8'),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=180) as response: return {'question':q, **json.load(response)}
 except Exception as exc: return {'question':q,'error':str(exc)}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(ask,questions))
Path('tests/http_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
for i,r in enumerate(results,1):
 print(f"{i}. {r['question']}\n{r.get('answer',r.get('error'))}\nИсточники: {', '.join(s['document']+' / '+s['section'] for s in r.get('sources',[]))}\nПоиск: "+str({k: len(v) for k,v in r.get('diagnostics',{}).items()})+'\n')
