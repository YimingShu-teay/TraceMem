import openai
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResponse:
    """Embedding response data model"""
    embeddings: List[List[float]]
    usage: Dict[str, Any]
    model: str
    response_time: float


class Embedding:
    """Embedding vector client using OpenAI API"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = "", model: str = "text-embedding-3-small"):
        """
        Initialize embedding client
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url)

        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30.0
        self.batch_size = 100 
        
        # Embedding dimension
        self.embedding_dim = self._get_embedding_dimension()
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if "text-embedding-3-small" in self.model:
            return 1536
        elif "text-embedding-3-large" in self.model:
            return 3072
        elif "text-embedding-ada-002" in self.model:
            return 1536
        else:
            return 1536
    
    def embed_texts(self, texts: List[str]) -> 'EmbeddingResponse':
        """
        Generate embedding vectors via OpenAI API
        """
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                usage={},
                model=self.model,
                response_time=0.0
            )
        
        start_time = time.time()
        all_embeddings = []
        total_usage = {"prompt_tokens": 0, "total_tokens": 0}
        
        for i in range(0, len(texts), self.batch_size):
            batch = [str(t).replace("\n", " ") for t in texts[i:i + self.batch_size]]
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        timeout=self.timeout)

                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    if response.usage:
                        usage = response.usage
                        total_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
                        total_usage["total_tokens"] += getattr(usage, "total_tokens", 0)
                    break 
                    
                except Exception as e:
                    print(f"Embedding API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                    else:
                        raise e
        
        response_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            usage=total_usage,
            model=self.model,
            response_time=response_time)
    
    def embed_text(self, text: str) -> List[float]:
        response = self.embed_texts([text])
        return response.embeddings[0] if response.embeddings else []
    