from ..configs.client import Client
from ..configs.embedding import Embedding
from ..configs.config import MemoryConfig
from ..storage.episode import Episode
from ..storage.semantic import SemanticMemory
from ..storage.experience import ExperienceMemory
from .segmenter import TopicSegmentor
from .categorizer import Categorizer
from .persona_extractor import PersonaExtractor
from .summarizer import Summarizer
# from configs.bm25 import BM25Search
# from cache.redis_manager import MemoryRedisManager
from .agentic_search import AgentReason
from typing import Dict, List
import threading
import json
import os

import logging
logger = logging.getLogger(__name__)


class TraceMem:
    _GLOBAL_DB_LOCK = threading.Lock()
    _SHARED_CHROMA_INDEX = None
    _SHARED_BACKEND = None


    def __init__(self):
        self._search_engine = None
        self._backend = None
        self._chroma_index = None
        self._db_lock = threading.RLock()
        self.config = MemoryConfig()  
        self.llm_client = Client(api_key=self.config.openai_api_key, 
                                 base_url= self.config.base_url,
                                 model=self.config.llm_model)
        self.embedding_client = Embedding(api_key=self.config.openai_api_key,
                                          base_url= self.config.base_url,
                                          model=self.config.embedding_model)
        self.topic_segmentor = TopicSegmentor(llm_client=self.llm_client)
        self.clusterer = Categorizer(backend=self.backend, config=self.config, llm_client=self.llm_client)
        self.extrator = PersonaExtractor(llm_client=self.llm_client)
        self.summarizer = Summarizer(llm_client=self.llm_client)
        # self.redis_manager = MemoryRedisManager()
        self.reason_agent = AgentReason(llm_client=self.llm_client,config=self.config,backend=self.backend)
    
    # @property
    # def search_engine(self):   
    #     if self._search_engine is None:       
    #         self._search_engine = BM25Search()
    #     return self._search_engine
        
    @property
    def backend(self):
        if TraceMem._SHARED_BACKEND is None:
            with TraceMem._GLOBAL_DB_LOCK:
                if TraceMem._SHARED_BACKEND is None:
                    print("Detected first access. Initializing GLOBAL ChromaEngine...")
                    from ..configs.chroma import ChromaEngine
                    TraceMem._SHARED_BACKEND = ChromaEngine(
                        embedding_client=self.embedding_client, 
                        config=self.config
                    )
                else:
                    print("Backend already initialized by another instance.")
        
        return TraceMem._SHARED_BACKEND

    @property
    def chroma_client(self):
        if TraceMem._SHARED_CHROMA_INDEX is None:
            with TraceMem._GLOBAL_DB_LOCK:
                if TraceMem._SHARED_CHROMA_INDEX is None:
                    from ..configs.chroma import ChromaIndex
                    TraceMem._SHARED_CHROMA_INDEX = ChromaIndex(backend=self.backend)
        return TraceMem._SHARED_CHROMA_INDEX

    def create_episode_memory(self,
                              roles: str,
                              time_stamp: str,
                              topic: Dict) -> Episode:
        # create one episodic memory
        episode_memory = Episode(
            user_id=roles,
            timestamp=time_stamp,
            summary=topic['summary'])
        return episode_memory
    
    def create_semantic_memory(self, 
                               roles: str,
                               semantic_memories: List[str],
                               time_stamp: str,
                               episode_id: str) -> List[SemanticMemory]:
        # create semantic memories
        all_semantic_memories = []
        for speaker, semantic_memory in semantic_memories.items():
            for content in semantic_memory:
                semantic = SemanticMemory(
                    content=content,
                    user_id=f"{roles}_{speaker}",
                    timestamp=time_stamp,
                    source_episode=episode_id)
                all_semantic_memories.append(semantic)
        return all_semantic_memories
    
    def create_experience_memory(self,
                                 roles: str,
                                 experiences: Dict,
                                 time_stamp: str,
                                 episode_id: str) -> List[ExperienceMemory]:
        # create personal experience for each user
        all_experiences = []
        for person, experience in experiences.items():
            exper = experience['Experience']
            if exper != "N/A":
                experience_memory = ExperienceMemory(
                    content=exper,
                    user_id=f"{roles}_{person}",
                    source_episode=episode_id,
                    timestamp=time_stamp)  
                all_experiences.append(experience_memory)
        return all_experiences


    def add_session(self, 
                    topics: List[Dict],
                    roles: str,
                    time_stamp: str) -> None:
        for topic in topics:
            self._process_single_topic(topic, roles, time_stamp)

    def _process_single_topic(self, 
                              topic: Dict, 
                              roles: str, 
                              time_stamp: str) -> None:
        # create one episode memory
        episode_memory = self.create_episode_memory(
                              roles=roles,
                              time_stamp=time_stamp,
                              topic=topic)
        
        # get episode id     
        episode_id = episode_memory.episode_id

        # create semantic list
        all_semantic_memories = self.create_semantic_memory(
                        roles=roles,
                        semantic_memories=topic['semantic_memories'],
                        time_stamp=time_stamp,
                        episode_id=episode_id)
        
        # create experience list
        all_experiences = self.create_experience_memory(
            roles=roles,
            experiences=topic['experience'],
            time_stamp=time_stamp,
            episode_id=episode_id)
            
        # save to chromadb, here needs a write block
        self.chroma_client.add_episode(roles=roles, episode=episode_memory)
        self.chroma_client.add_semantic(memory=all_semantic_memories)
        self.chroma_client.add_experience(memory=all_experiences)

        # save to redis cache (if you want to use cache or you want to use BM25 search)
        # self.redis_manager.save_episode(
        #     collection_name=f"triggmem_{roles}_episodes",
        #     episode_id=episode_memory.episode_id, 
        #     episode_data=episode_memory)        
        # self.redis_manager.save_semantic_memory( semantic_data=all_semantic_memories)
    
    def add_memories(self, 
                     sessions: List[Dict], 
                     roles: str) -> None:
        for key, value in sessions.items(): 
            time_stamp = sessions.get(key, [{}])[0].get('metadata', {}).get('dataset_timestamp')
            topics = self.topic_segmentor.topic_segment_session(messages=value)
            topics = self.summarizer.episodes_summary(topics=topics, mesaages=value)
            topics = self.extrator.experiences_extraction(topics=topics)
            self.add_session(topics=topics, roles=roles, time_stamp=time_stamp)


    def build_personal_card(self,
                            user_id: str,
                            roles: str) -> None:
        
        # create and summarize topics
        topic_clusters, topics = self.clusterer.topic_categorize(roles=roles,
                                                                 user_id=user_id)

        # create and summarize threads
        topics = self.clusterer.thread_categorize(topic_clusters=topic_clusters, 
                                                  topics=topics)  

        # save user experience card      
        card = json.dumps(topics, ensure_ascii=False, indent=2)
        if not os.path.exists(self.config.cards_dir):
            os.makedirs(self.config.cards_dir, exist_ok=True)
        with open(os.path.join(self.config.cards_dir,f"{roles}_{user_id}.json"), 'w', encoding='utf-8') as f:
            f.write(card)      

        thread_map = json.dumps(self.clusterer.thread_map, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config.cards_dir,f"{roles}_{user_id}_thread_map.json"), 'w', encoding='utf-8') as f:
            f.write(thread_map)      

    def build_personal_cards(self,
                             speaker_a: str,
                             speaker_b: str):
        # build personal cards
        roles = f"{speaker_a}_{speaker_b}"
        self.build_personal_card(user_id=speaker_b,
                                 roles=roles)    
        self.build_personal_card(user_id=speaker_a,
                                 roles=roles)  

    
    def answer(self, 
               question: str, 
               speakers: List[str]) -> str:

        response = self.reason_agent.answer(question=question,
                                            speakers=speakers)
        return response
