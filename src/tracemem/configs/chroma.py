import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from ..storage.episode import Episode
from ..storage.semantic import SemanticMemory
from ..storage.experience import ExperienceMemory
from ..storage.thread import ThreadMemory
from .embedding import Embedding
from .config import MemoryConfig
from abc import ABC, abstractmethod
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

class ChromaEngine:
    def __init__(self, embedding_client: Embedding, config: MemoryConfig):
        self.embedding_client = embedding_client
        self.config = config
        
        self.persist_directory = config.chroma_persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, 
                              allow_reset=True))
        
        self.collection_prefix = config.chroma_collection_prefix
        
        self._collection_locks = defaultdict(threading.RLock)
        
    
    def _get_collection_lock(self, collection_name: str) -> threading.RLock:
        return self._collection_locks[collection_name]
    
    def _get_episode_collection_name(self, user_id: str) -> str:
        return f"{self.collection_prefix}_{user_id}_episodes"
    
    def _get_semantic_collection_name(self, user_id: str) -> str:
        return f"{self.collection_prefix}_{user_id}_semantic"
    
    def _get_experience_collection_name(self, user_id: str) -> str:
        return f"{self.collection_prefix}_{user_id}_experience"

    def _get_thread_collection_name(self, user_id: str) -> str:
        return f"{self.collection_prefix}_{user_id}_thread"
        
    def _get_episode_collection(self, user_id: str):
        collection_name = self._get_episode_collection_name(user_id)

        with self._get_collection_lock(collection_name):
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.debug(f"Episode Collection: {collection_name}")
            except (ValueError, Exception) as e:
                try:
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "episodes"}
                    )
                    logger.info(f"Create new Episode Collection: {collection_name}")
                except Exception as create_error:
                    logger.error(f"Create Episode Set Fail: {create_error}")
                    try:
                        self.client.delete_collection(name=collection_name)
                    except:
                        pass
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "episodes"}
                    )
                    logger.info(f"Recreate Episode Collection: {collection_name}")
            
        
            return collection
    
    def _get_semantic_collection(self, user_id: str):
        collection_name = self._get_semantic_collection_name(user_id)
        with self._get_collection_lock(collection_name):
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.debug(f"Get existing Semantic Set: {collection_name}")
            except (ValueError, Exception) as e:
                try:
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "semantic"}
                    )
                    logger.info(f"Create new Semantic Set: {collection_name}")
                except Exception as create_error:
                    logger.error(f"Create Semantic Set Fail: {create_error}")
                    try:
                        self.client.delete_collection(name=collection_name)
                    except:
                        pass
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "semantic"}
                    )
                    logger.info(f"Recreating Semantic collection: {collection_name}")
        
            return collection
    

    def _get_experience_collection(self, user_id: str):
        collection_name = self._get_experience_collection_name(user_id)

        with self._get_collection_lock(collection_name):
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.debug(f"Retrieved existing Experience collection: {collection_name}")
            except (ValueError, Exception) as e:
                try:
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "experiences"}
                    )
                    logger.info(f"Creating new Experience collection: {collection_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create Experience collection: {create_error}")
                    try:
                        self.client.delete_collection(name=collection_name)
                    except:
                        pass
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "experiences"}
                    )
                    logger.info(f"Recreating Experience collection: {collection_name}")
            
        
            return collection
    
    def _get_experience_collection(self, user_id: str):
        collection_name = self._get_experience_collection_name(user_id)
        with self._get_collection_lock(collection_name):
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.debug(f"Retrieved existing Experience collection: {collection_name}")
            except (ValueError, Exception) as e:
                try:
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "experiences"})
                    logger.info(f"Creating new Experience collection: {collection_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create Experience collection: {create_error}")
                    try:
                        self.client.delete_collection(name=collection_name)
                    except:
                        pass
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "experiences"})
                    logger.info(f"Recreating Experience collection: {collection_name}")
        
            return collection
    

    def _get_thread_collection(self, user_id: str):
        collection_name = self._get_thread_collection_name(user_id)

        with self._get_collection_lock(collection_name):
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.debug(f"Retrieved existing Thread collection: {collection_name}")
            except (ValueError, Exception) as e:
                try:
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "thread"})
                    
                    logger.info(f"Creating new Thread collection: {collection_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create Thread collection: {create_error}")
                    try:
                        self.client.delete_collection(name=collection_name)
                    except:
                        pass
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "thread"})
                    
                    logger.info(f"Recreating Thread collection: {collection_name}")
            
            return collection

    
    def add_episode(self, user_id: str, episode: Episode):
        collection_name = self._get_episode_collection_name(user_id)
        with self._get_collection_lock(collection_name):
            try:
                collection = self._get_episode_collection(user_id)
                
                existing = collection.get(ids=[episode.episode_id])
                if existing['ids']:
                    logger.debug(f"Episode {episode.episode_id} exists, ignore")
                    return
                
                embed_text = f"{episode.summary}"
                metadata = {
                    "episode_id": episode.episode_id,
                    "user_id": episode.user_id,
                    "created_at": episode.created_at.isoformat(),
                    "timestamp": episode.timestamp,
                    "type": "episode"
                }
                
                embedding_response = self.embedding_client.embed_texts([embed_text])
                embedding = embedding_response.embeddings[0]

                collection.add(
                    ids=[episode.episode_id],
                    documents=[episode.summary],
                    metadatas=[metadata],
                    embeddings=[embedding]
                )
                
                logger.debug(f"add Episode {episode.episode_id} to user {user_id}")
                
            except Exception as e:
                logger.error(f"add Episode {episode.episode_id} to user {user_id} Error: {e}")
                raise
    
    def add_semantic_memory(self, user_id: str, memory: SemanticMemory):
        collection_name = self._get_semantic_collection_name(user_id)
        with self._get_collection_lock(collection_name):
            try:
                collection = self._get_semantic_collection(user_id)
                
                existing = collection.get(ids=[memory.memory_id])
                if existing['ids']:
                    logger.debug(f"SemanticMemory {memory.memory_id} already exists, skipping addition")
                    return
                
                metadata = {
                    "memory_id": memory.memory_id,
                    "user_id": memory.user_id,
                    "created_at": memory.created_at.isoformat(),
                    "source_episode": memory.source_episode,  
                    "revision_count": memory.revision_count,
                    "type": "semantic"
                }
                
                if memory.updated_at:
                    metadata["updated_at"] = memory.updated_at.isoformat()
                
                embedding_response = self.embedding_client.embed_texts([memory.content])
                embedding = embedding_response.embeddings[0]

                collection.add(
                    ids=[memory.memory_id],
                    documents=[memory.content],
                    metadatas=[metadata],
                    embeddings=[embedding]
                )

                logger.debug(f"SemanticMemory {memory.memory_id} to user {user_id}")
                
            except Exception as e:
                logger.error(f"SemanticMemory {memory.memory_id} to user {user_id} Error: {e}")
                raise
        
    def add_semantic_memories(self,semantic_memories:List[SemanticMemory]):

        for memory in semantic_memories:
            self.add_semantic_memory(user_id=memory.user_id,
                                     memory=memory)
            

    def add_experience_memory(self, user_id: str, experience: ExperienceMemory):
        collection_name = self._get_experience_collection_name(user_id)
        with self._get_collection_lock(collection_name):
            try:
                collection = self._get_experience_collection(user_id)
                
                existing = collection.get(ids=[experience.experience_id])
                if existing['ids']:
                    logger.debug(f"Episode {experience.experience_id} exists, ignore")
                    return
                
                embed_text = f"Keywords:{experience.content}"
                metadata = {
                    "experience_id": experience.memory_id,
                    "user_id": experience.user_id,
                    "created_at": experience.created_at.isoformat(),
                    "timestamp": experience.timestamp,
                    "source_episode": experience.source_episode, 
                    "type": "experience"
                }
                
                embedding_response = self.embedding_client.embed_texts([embed_text])
                embedding = embedding_response.embeddings[0]

                collection.add(
                    ids=[experience.memory_id],
                    documents=[experience.content],
                    metadatas=[metadata],
                    embeddings=[embedding]
                )
                
                logger.debug(f"add Experience {experience.memory_id} to user {user_id}")
                
            except Exception as e:
                logger.error(f"add Experience {experience.memory_id} to user {user_id} Error: {e}")
                raise

    def add_experience_memories(self,experience_memories:List[ExperienceMemory]):
        
        for memory in experience_memories:
            self.add_experience_memory(user_id=memory.user_id,
                                     experience=memory)
            

    def add_thread_memory(self, user_id: str, thread: ThreadMemory):
        collection_name = self._get_thread_collection_name(user_id)

        with self._get_collection_lock(collection_name):
            try:
                collection = self._get_thread_collection(user_id)
                
                existing = collection.get(ids=[thread.thread_id])
                if existing['ids']:
                    logger.debug(f"Thread {thread.thread_id} exists, ignore")
                    return
                
                embed_text = f"{thread.content}"
                metadata = {
                    "source_episode": thread.source_episode,
                    "user_id": thread.user_id,
                    "created_at": thread.created_at.isoformat(),
                    "type": "thread"}
                
                embedding_response = self.embedding_client.embed_texts([embed_text])
                embedding = embedding_response.embeddings[0]

                collection.add(
                    ids=[thread.thread_id],
                    documents=[thread.content],
                    metadatas=[metadata],
                    embeddings=[embedding])
                
                logger.debug(f"add Thread {thread.thread_id} to user {user_id}")
                
            except Exception as e:
                logger.error(f"add Thread {thread.thread_id} to user {user_id} Error: {e}")
                raise

    
    def search_episodes(self, user_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        
        collection_name = self._get_episode_collection_name(user_id)
        with self._get_collection_lock(collection_name):
            try:
                collection = self._get_episode_collection(user_id)
                
                if collection.count() == 0:
                    logger.debug(f"Episode collection is empty for user {user_id}")
                    return []
                
                query_embedding = self.embedding_client.embed_text(query)
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, collection.count())
                )
                
                search_results = []
                for i, doc_id in enumerate(results['ids'][0]):
                    result = {
                        "episode_id": doc_id,
                        "title": results['metadatas'][0][i].get('title', ''),
                        "content": results['documents'][0][i],
                        "distance": results['distances'][0][i],
                        "score": 1 - results['distances'][0][i],  
                        "metadata": results['metadatas'][0][i],
                        "type": "episode"
                    }
                    search_results.append(result)
                
                logger.debug(f"Search for user {user_id} episodes returned {len(search_results)} results")
                return search_results
                
            except Exception as e:
                logger.error(f"Failed to search episodes for user {user_id}: {e}")
                return []
        
    def search_semantic_memories(self, user_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        collection_name = self._get_semantic_collection_name(user_id)
        with self._get_collection_lock(collection_name):
            try:
                collection = self._get_semantic_collection(user_id)
                
                if collection.count() == 0:
                    logger.debug(f"SemanticMemory collection is empty for user {user_id}")
                    return []
                
                query_embedding = self.embedding_client.embed_text(query)
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, collection.count())
                )
                
                search_results = []
                for i, doc_id in enumerate(results['ids'][0]):
                    result = {
                        "memory_id": doc_id,
                        "content": results['documents'][0][i],
                        "knowledge_type": results['metadatas'][0][i].get('knowledge_type', ''),
                        "distance": results['distances'][0][i],
                        "score": 1 - results['distances'][0][i],  
                        "metadata": results['metadatas'][0][i],
                        "type": "semantic"
                    }
                    search_results.append(result)
                
                logger.debug(f"Search for user {user_id} SemanticMemory returned {len(search_results)} results")
                return search_results
                
            except Exception as e:
                logger.error(f"Failed to search SemanticMemory for user {user_id}: {e}")
                return []


    def search_experiences(self, user_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        
        collection_name = self._get_experience_collection_name(user_id)
        with self._get_collection_lock(collection_name):
            try:
                collection = self._get_experience_collection(user_id)
                
                if collection.count() == 0:
                    logger.debug(f"Experience collection is empty for user {user_id}")
                    return []
                
                query_embedding = self.embedding_client.embed_text(query)
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, collection.count())
                )
                
                search_results = []
                for i, doc_id in enumerate(results['ids'][0]):
                    result = {
                        "experience_id": doc_id,
                        "title": results['metadatas'][0][i].get('title', ''),
                        "content": results['documents'][0][i],
                        "distance": results['distances'][0][i],
                        "score": 1 - results['distances'][0][i], 
                        "metadata": results['metadatas'][0][i],
                        "type": "experience"
                    }
                    search_results.append(result)
                
                logger.debug(f"Search for user {user_id} experiences returned {len(search_results)} results")
                return search_results
                
            except Exception as e:
                logger.error(f"Failed to search experiences for user {user_id}: {e}")
                return []
            
    def search_thread_memories(self, user_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:

        collection_name = self._get_thread_collection_name(user_id)
        with self._get_collection_lock(collection_name):
            try:
                collection = self._get_thread_collection(user_id)
                
                if collection.count() == 0:
                    logger.debug(f"ThreadMemory collection is empty for user {user_id}")
                    return []
                
                query_embedding = self.embedding_client.embed_text(query)
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, collection.count())
                )
                
                search_results = []
                for i, doc_id in enumerate(results['ids'][0]):

                    result = {
                        "memory_id": doc_id,
                        "content": results['documents'][0][i],
                        "source_episode": results['metadatas'][0][i].get('source_episode', ''),
                        "knowledge_type": results['metadatas'][0][i].get('knowledge_type', ''),
                        "distance": results['distances'][0][i],
                        "score": 1 - results['distances'][0][i],  
                        "metadata": results['metadatas'][0][i],
                        "type": "thread"
                    }
                    search_results.append(result)

                logger.debug(f"Search for user {user_id} ThreadMemory returned {len(search_results)} results")
                return search_results
                
            except Exception as e:
                logger.error(f"Error fetching ThreadMemory for user {user_id}: {e}")
                return []
        

class VectorIndex(ABC):
    """Vector index abstraction for episodic/semantic retrieval."""

    @abstractmethod
    def add_episode(self, user_id: str, episode: Episode, embedding: Optional[List[float]] = None) -> None: ...

    @abstractmethod
    def add_semantic(self, user_id: str, memory: SemanticMemory, embedding: Optional[List[float]] = None) -> None: ...

    @abstractmethod
    def search_episodes(self, user_id: str, query: str, top_k: int) -> List[Dict]: ...

    @abstractmethod
    def search_semantics(self, user_id: str, query: str, top_k: int) -> List[Dict]: ...


class ChromaIndex(VectorIndex):
    def __init__(self, backend: ChromaEngine) -> None:
        self._backend = backend

    def add_episode(self, roles: str, episode: Episode) -> None:
        self._backend.add_episode(roles, episode)

    def add_semantic(self,memory: List[SemanticMemory]) -> None:
        self._backend.add_semantic_memories(memory)

    def add_experience(self,memory: List[ExperienceMemory]) -> None:
        self._backend.add_experience_memories(memory)

    def add_thread(self,memory: ThreadMemory) -> None:
        self._backend.add_thread_memory(memory)

    def search_episodes(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        return self._backend.search_episodes(user_id, query, top_k)

    def search_semantics(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        return self._backend.search_semantic_memories(user_id, query, top_k)

    def search_threads(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        return self._backend.search_thread_memories(user_id, query, top_k)