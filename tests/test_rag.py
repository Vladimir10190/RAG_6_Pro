"""Local checks for hybrid retrieval and source validation; no network needed."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import faiss
import numpy as np
from app import rag, main

class FakeClient:
    def __init__(self, **kwargs):
        self.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])]))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'answer': 'made up', 'source_ids': ['S999']})))])))

class RagChecks(unittest.TestCase):
    def test_hybrid_and_stale_documents(self):
        temp_root = Path(__file__).resolve().parents[1] / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            kb = root / 'kb'; kb.mkdir()
            (kb / 'a.md').write_text('# API\n\n## Лимиты\nHTTP 429 Retry-After', encoding='utf8')
            chunks = [{'text': 'HTTP 429 Retry-After', 'document': 'API', 'section': 'Лимиты', 'file': 'a.md'}]
            idx = faiss.IndexFlatIP(2); idx.add(np.array([[1, 0]], dtype='float32'))
            index = root / 'index.faiss'
            with index.open('wb') as f: faiss.write_index(idx, faiss.PyCallbackIOWriter(f.write))
            manifest = root / 'chunks.json'; manifest.write_text(json.dumps({'model':'test', 'chunks':chunks}), encoding='utf8')
            with patch.multiple(rag, KB_DIR=kb, INDEX_PATH=index, CHUNKS_PATH=manifest), patch.dict(os.environ, {'OPENAI_EMBEDDING_MODEL':'test'}):
                result = rag.Retriever().search(FakeClient(), 'HTTP 429', 4, 0.25)
                self.assertEqual([result[key][0]['id'] for key in ('vector','bm25','context')], [0,0,0])
                self.assertEqual(len(result['context']), 1)
                (kb / 'a.md').write_text('# API\n\n## Лимиты\nИзменено', encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'индекс'):
                    rag.Retriever()

    def test_invalid_citations_are_rejected(self):
        fragment = {'id':0, 'document':'Doc', 'section':'Sec', 'file':'a.md', 'text':'text', 'rank':1, 'score':1.0}
        fake_retriever = SimpleNamespace(search=lambda *a: {'vector':[fragment], 'bm25':[fragment], 'context':[fragment]})
        with patch.object(main, 'Retriever', return_value=fake_retriever), patch.object(main, 'OpenAI', FakeClient), patch.dict(os.environ, {'OPENAI_API_KEY':'test'}):
            response = main.chat(main.ChatRequest(question='test'))
        self.assertTrue(response.no_answer)
        self.assertEqual(response.sources, [])

class KnowledgeUpdates(unittest.TestCase):
    def test_upload_replace_reindex_and_failure(self):
        temp_root = Path(__file__).resolve().parents[1] / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory); kb = root / 'kb'; kb.mkdir(); data = root / 'data'; data.mkdir()
            class Embeddings:
                def create(self, **kwargs):
                    return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in kwargs['input']])
            client = SimpleNamespace(embeddings=Embeddings())
            with patch.multiple(rag, KB_DIR=kb, DATA_DIR=data, INDEX_PATH=data/'index.faiss', CHUNKS_PATH=data/'chunks.json'), patch.object(main, 'KB_DIR', kb), patch.object(main, 'OpenAI', return_value=client), patch.dict(os.environ, {'OPENAI_API_KEY':'test', 'OPENAI_EMBEDDING_MODEL':'test'}):
                result = main.upload_document(main.DocumentUpload(filename='new.md', content='# Документ\n\n## Правило\nСтарый код ABC123'))
                self.assertFalse(result['replaced'])
                self.assertEqual(main.reindex()['chunks'], 1)
                self.assertEqual(main.knowledge_status()['index_state'], 'ready')
                self.assertIn('Старый код', rag.Retriever().search(client, 'ABC123', 1, 0)['context'][0]['text'])
                before = (data/'chunks.json').read_bytes()
                result = main.upload_document(main.DocumentUpload(filename='new.md', content='# Документ\n\n## Правило\nНовый код XYZ999'))
                self.assertTrue(result['replaced'])
                self.assertEqual(main.knowledge_status()['index_state'], 'stale')
                with self.assertRaises(ValueError): rag.Retriever()
                with patch.object(client.embeddings, 'create', side_effect=RuntimeError('provider failed')):
                    with self.assertRaises(main.HTTPException): main.reindex()
                self.assertEqual((data/'chunks.json').read_bytes(), before)
                self.assertEqual(main.reindex()['chunks'], 1)
                found = rag.Retriever().search(client, 'XYZ999', 1, 0)
                self.assertIn('Новый код', found['context'][0]['text'])
                self.assertNotIn('Старый код', found['context'][0]['text'])
                self.assertEqual(found['bm25'][0]['file'], 'new.md')
                self.assertEqual(main.knowledge_status()['index_state'], 'ready')
                with self.assertRaises(main.HTTPException):
                    main.upload_document(main.DocumentUpload(filename='../escape.md', content='test'))

if __name__ == '__main__': unittest.main()
