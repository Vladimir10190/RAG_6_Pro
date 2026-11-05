const form = document.querySelector('#form');
const input = document.querySelector('#question');
const messages = document.querySelector('#messages');
const send = document.querySelector('#send');

function addMessage(text, role, sources = [], diagnostics = null) {
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  article.append(bubble);
  if (sources.length) {
    const box = document.createElement('div');
    box.className = 'sources';
    const label = document.createElement('strong');
    label.textContent = 'Источники: ';
    box.append(label);
    for (const source of sources) {
      const tag = document.createElement('a');
      tag.href = `/api/documents/${encodeURIComponent(source.file)}`;
      tag.target = '_blank';
      tag.rel = 'noopener';
      tag.className = 'source';
      tag.textContent = `${source.document} · ${source.section}`;
      box.append(tag);
    }
    article.append(box);
  }
  if (diagnostics) {
    const details = document.createElement('details');
    details.className = 'diagnostics';
    const summary = document.createElement('summary');
    summary.textContent = 'Диагностика поиска';
    details.append(summary);
    for (const [key, title] of [['vector', 'Векторный поиск'], ['bm25', 'BM25'], ['context', 'Итоговый контекст LLM (RRF)']]) {
      const heading = document.createElement('h4');
      heading.textContent = `${title}: ${diagnostics[key].length}`;
      details.append(heading);
      for (const item of diagnostics[key]) {
        const block = document.createElement('div');
        block.className = 'fragment';
        const label = document.createElement('strong');
        label.textContent = `#${item.rank} · S${item.id} · ${item.document} — ${item.section} · ${item.score.toFixed(4)}`;
        const content = document.createElement('div');
        content.textContent = item.text;
        block.append(label, content);
        details.append(block);
      }
    }
    article.append(details);
  }
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addMessage(question, 'user');
  input.value = '';
  send.disabled = true;
  const pending = addMessage('Ищу ответ в документах…', 'assistant');
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Не удалось получить ответ.');
    pending.remove();
    addMessage(result.answer, 'assistant', result.sources, result.diagnostics);
  } catch (error) {
    pending.remove();
    addMessage(error.message || 'Ошибка соединения с сервером.', 'assistant');
  } finally {
    send.disabled = false;
    input.focus();
  }
});

input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
});

const knowledgeStatus = document.querySelector('#knowledge-status');
const documentList = document.querySelector('#document-list');
const documentFile = document.querySelector('#document-file');
const uploadButton = document.querySelector('#upload-document');
const reindexButton = document.querySelector('#reindex');
let working = false;

async function refreshKnowledge() {
  try {
    const response = await fetch('/api/knowledge');
    if (!response.ok) throw new Error('Не удалось загрузить состояние базы.');
    const state = await response.json();
    documentList.replaceChildren();
    for (const documentInfo of state.documents) {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = `/api/documents/${encodeURIComponent(documentInfo.filename)}`;
      link.target = '_blank'; link.rel = 'noopener';
      link.textContent = documentInfo.filename;
      item.append(link); documentList.append(item);
    }
    knowledgeStatus.textContent = state.indexing ? 'Идёт обновление базы…' : state.message;
    knowledgeStatus.dataset.state = state.index_state;
    uploadButton.disabled = reindexButton.disabled = working || state.indexing;
  } catch (error) { knowledgeStatus.textContent = error.message; }
}

async function knowledgeAction(url, options, busyText) {
  working = true; uploadButton.disabled = reindexButton.disabled = true;
  knowledgeStatus.textContent = busyText;
  try {
    const response = await fetch(url, options);
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Не удалось выполнить операцию.');
    await refreshKnowledge();
    knowledgeStatus.textContent = result.message;
    return true;
  } catch (error) { knowledgeStatus.textContent = error.message; return false; }
  finally { working = false; uploadButton.disabled = reindexButton.disabled = false; }
}

uploadButton.addEventListener('click', async () => {
  const file = documentFile.files[0];
  if (!file) { knowledgeStatus.textContent = 'Выберите Markdown-файл.'; return; }
  if (!file.name.endsWith('.md') || file.size > 1000000) { knowledgeStatus.textContent = 'Нужен файл .md размером до 1 МБ.'; return; }
  try {
    const content = new TextDecoder('utf-8', {fatal: true}).decode(await file.arrayBuffer());
    const ok = await knowledgeAction('/api/knowledge/documents', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({filename:file.name, content})}, 'Сохраняю документ…');
    if (ok) documentFile.value = '';
  } catch (_) { knowledgeStatus.textContent = 'Не удалось прочитать файл в кодировке UTF-8.'; }
});
reindexButton.addEventListener('click', () => knowledgeAction('/api/knowledge/reindex', {method: 'POST'}, 'Создаю embeddings и обновляю индекс. Это может занять некоторое время…'));
refreshKnowledge();
