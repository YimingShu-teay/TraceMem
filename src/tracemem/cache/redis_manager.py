from typing import Any, Optional, List
from .redis_processor import RedisDataProcessor 
from .redis_provider import RedisProvider
import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def get_logger(name: str):
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"), 
        maxBytes=10*1024*1024,  
        backupCount=5           
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = get_logger(__name__)

class MemoryRedisManager:
    def __init__(self, max_length: int = 500):
        self.redis_provider = RedisProvider()
        self.max_length = max_length
        self.processor = RedisDataProcessor()
        self.default_ttl = 7 * 24 * 3600

    def _save_to_hash_collection(self, collection_name: str, item_id: str, data: Any):
            try:
                client = self.redis_provider.get_client()
                storable_data = self.processor.process_data_for_storage(data)
                
                client.hset(collection_name, item_id, storable_data)
                client.expire(collection_name, self.default_ttl)
                
                logger.info(f"Success: Saved to collection [{collection_name}], Key: {item_id}")
            except Exception as e:
                logger.error(f"Error saving to collection {collection_name}: {e}")

    def _save_semantic_to_hash_collection(self, semantic_data: Any):
            try:
                client = self.redis_provider.get_client()
                
                memories = semantic_data if isinstance(semantic_data, list) else [semantic_data]
                if not memories:
                    return

                grouped_memories = {}
                for memory in memories:
                    u_id = getattr(memory, 'user_id', getattr(memory, 'roles', 'default'))
                    if u_id not in grouped_memories:
                        grouped_memories[u_id] = []
                    grouped_memories[u_id].append(memory)

                for u_id, m_list in grouped_memories.items():
                    collection_name = f"triggmem_{u_id}_semantics"
                    pipe = client.pipeline()
                    
                    for memory in m_list:
                        storable_data = self.processor.process_data_for_storage(memory)
                        pipe.hset(collection_name, memory.memory_id, storable_data)
                    
                    pipe.expire(collection_name, self.default_ttl)
                    pipe.execute()
                    logger.info(f"Success: Saved {len(m_list)} memories to [{collection_name}].")

            except Exception as e:
                logger.error(f"Error saving grouped semantic memories: {e}")
    
    # --- Specialized Save Methods ---

    def save_cluster(self, collection_name: str, cluster_id: str, cluster_data: Any):
        self._save_to_hash_collection(collection_name, cluster_id, cluster_data)

    def save_episode(self, collection_name: str, episode_id: str, episode_data: Any):
        self._save_to_hash_collection(collection_name, episode_id, episode_data)

    def save_semantic_memory(self,semantic_data: Any):
        self._save_semantic_to_hash_collection(semantic_data)

    def get_cluster_by_id(self, collection_name: str, cluster_id: str) -> Optional[Any]:

        try:
            client = self.redis_provider.get_client()
            
            raw_data = client.hget(collection_name, cluster_id)
            
            if not raw_data:
                logger.warning(f"ID {cluster_id} not found in Redis collection [{collection_name}]")
                return None
            
            processed = self.processor.process_data_from_storage(raw_data)
            return processed.get("data")
        except Exception as e:
            logger.error(f"Redis retrieval failed (Coll: {collection_name}, ID: {cluster_id}): {str(e)}")
            return None

    def get_all_clusters_in_collection(self, collection_name: str) -> List[Any]:

        try:
            client = self.redis_provider.get_client()

            all_data = client.hgetall(collection_name)
            
            results = []
            for cid_bytes, data_bytes in all_data.items():
                processed = self.processor.process_data_from_storage(data_bytes)
                if processed.get("data"):
                    results.append(processed["data"])
            return results
        except Exception as e:
            logger.error(f"Failed to fetch all data from collection [{collection_name}]: {str(e)}")
            return []

    def set_processing_lock(self, cluster_id: str, ttl: int = 60) -> bool:

        try:
            client = self.redis_provider.get_client()
            lock_key = f"mem:lock:cluster:{cluster_id}"
            result = client.set(lock_key, "locked", ex=ttl, nx=True)
            return result is True
        except Exception as e:
            logger.error(f"Failed to set lock (ID: {cluster_id}): {str(e)}")
            return False

    def delete_cluster_from_collection(self, collection_name: str, cluster_id: str):

        try:
            client = self.redis_provider.get_client()
            client.hdel(collection_name, cluster_id)
            logger.info(f"Deleted key {cluster_id} from Redis collection [{collection_name}]")
        except Exception as e:
            logger.error(f"Redis deletion failed (Coll: {collection_name}, ID: {cluster_id}): {str(e)}")

    def delete_entire_collection(self, collection_name: str):

        try:
            client = self.redis_provider.get_client()
            client.delete(collection_name)
            logger.info(f"Cleared entire Redis collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {collection_name}, {str(e)}")