import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import spacy
from spacy.lang.en import English
from spacy.lang.zh import Chinese
from cache.redis_manager import MemoryRedisManager
SPACY_AVAILABLE = True

logger = logging.getLogger(__name__)


class BM25Search:
    def __init__(self, language: str = "en"):
        self.language = language
        self.redis_manager = MemoryRedisManager()
        self.redis_provider = self.redis_manager.redis_provider  
        self.processor = self.redis_manager.processor
        self.nlp = self._initialize_spacy_tokenizer()
        
        logger.info(f"BM25 engine initialization complete. Language: {self.language}")

    def _initialize_spacy_tokenizer(self):
        if not SPACY_AVAILABLE: return None
        try:
            if self.language == "zh":
                try: nlp = spacy.load("zh_core_web_sm")
                except OSError: nlp = Chinese(); nlp.add_pipe('sentencizer')
            else:
                try: nlp = spacy.load("en_core_web_sm")
                except OSError: nlp = English(); nlp.add_pipe('sentencizer')
            return nlp
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {e}"); return None

    def _tokenize(self, text: str) -> List[str]:
        try:
            if self.nlp:
                doc = self.nlp(text.lower())
                return [t.lemma_ for t in doc if not t.is_punct and not t.is_space and not t.is_stop and len(t.text.strip()) > 1]
            import re
            tokens = re.findall(r'\b\w+\b', text.lower())
            return [t for t in tokens if len(t) > 1]
        except Exception as e:
            logger.error(f"Tokenization error: {e}"); return text.lower().split()

    def _get_data_from_redis(self, collection_name: str, target_ids: List[str] = None) -> List[Any]:
        try:
            client = self.redis_provider.get_client()
            if target_ids:
                raw_data_list = client.hmget(collection_name, target_ids)
                all_raw_data = {target_ids[i]: val for i, val in enumerate(raw_data_list) if val}
            else:
                all_raw_data = client.hgetall(collection_name)

            results = []
            for _, raw_val in all_raw_data.items():
                unpacked = self.processor.process_data_from_storage(raw_val)
                item_obj = unpacked.get("data")
                if item_obj: results.append(item_obj)
            return results
        except Exception as e:
            logger.error(f"Failed to extract Redis data [{collection_name}]: {e}"); return []

    def search_clusters(self, collection_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        clusters = self._get_data_from_redis(collection_name)
        if not clusters: return []

        valid_docs, filtered_corpus = [], []
        for c in clusters:
            tokens = self._tokenize(c.content)
            if tokens: filtered_corpus.append(tokens); valid_docs.append(c)
        
        if not filtered_corpus: return []
        bm25 = BM25Okapi(filtered_corpus)
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                c = valid_docs[idx]
                results.append({
                    "type": "cluster", "score": float(scores[idx]),
                    "cluster_id": c.cluster_id, "content": c.content,
                    "included_episodes": c.included_episode.split(",") if isinstance(c.included_episode, str) else c.included_episode,
                    "timestamp": c.timestamp.isoformat() if hasattr(c.timestamp, 'isoformat') else str(c.timestamp)
                })
        return results

    def search_episodes(self, collection_name: str, query: str, target_ids: List[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:

        episodes = self._get_data_from_redis(collection_name, target_ids)
        if not episodes: return []

        valid_docs, filtered_corpus = [], []
        for ep in episodes:
            tokens = self._tokenize(ep.summary) 
            if tokens: filtered_corpus.append(tokens); valid_docs.append(ep)
        
        if not filtered_corpus: return []
        bm25 = BM25Okapi(filtered_corpus)
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                ep = valid_docs[idx]
                results.append({
                    "type": "episodic", "score": float(scores[idx]),
                    "episode_id": ep.episode_id, "content": ep.summary,
                    "timestamp": ep.timestamp.isoformat() if hasattr(ep.timestamp, 'isoformat') else str(ep.timestamp)
                })
        return results

    def search_semantic_memories(self, collection_name: str, query: str, target_ep_id: str = None, top_k: int = 10) -> List[Dict[str, Any]]:

        memories = self._get_data_from_redis(collection_name)
        if not memories: return []

        if target_ep_id:
            memories = [m for m in memories if target_ep_id in m.source_episode]

        valid_docs, filtered_corpus = [], []
        for m in memories:
            tokens = self._tokenize(m.content)
            if tokens: filtered_corpus.append(tokens); valid_docs.append(m)
        
        if not filtered_corpus: 
            return []
        
        bm25 = BM25Okapi(filtered_corpus)
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                m = valid_docs[idx]
                results.append({
                    "type": "semantic", "score": float(scores[idx]),
                    "memory_id": m.memory_id, "content": m.content,
                    "related_episode": m.source_episode
                })
        return results

   
