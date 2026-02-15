from dataclasses import dataclass
from typing import Dict, Any
import os


@dataclass
class MemoryConfig:
    """Memory System Configuration"""

    # === Model Configuration ===
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # === Language Configuration ===
    language: str = "en"  # "en" for English, "zh" for Chinese
    
    # === Buffer Configuration ===
    buffer_size_min: int = 2       # Minimum buffer size
    buffer_size_max: int = 25       # Maximum buffer size
    
    # === Storage / Index Backends ===
    storage_backend: str = "filesystem"         # "filesystem" | "memory"
    vector_index_backend: str = "chroma"        # "chroma" | "memory"
    lexical_index_backend: str = "bm25"          # "bm25" | "memory"

    # === Vector Database Configuration ===
    vector_db_type: str = "chroma"              # Vector database type: "chroma"
    chroma_persist_directory: str = "./chroma_db"  # ChromaDB persistence directory
    chroma_collection_prefix: str = "tracemem"    # ChromaDB collection name prefix
    
    # === Performance Configuration ===
    batch_size: int = 32                        # Batch size
    max_workers: int = 4                        # Maximum number of worker threads
    semantic_generation_workers: int = 8         # Number of semantic memory generation threads
    
    # === Cache Configuration ===
    enable_cache: bool = True                   # Enable cache
    cache_size: int = 1000                      # Cache size
    cache_ttl_seconds: int = 3600               # Cache expiration time (seconds)
    semantic_cache_ttl: int = 600               # Semantic cache TTL
    episode_cache_ttl: int = 600                # Episode cache TTL
    
    # === Environment Variable Configuration ===
    # openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    base_url: str = os.getenv("BASE_URL")

    # save dir
    cards_dir: str = "./cards/"
    answers_dir: str = "./answers/"
    db_path: str = "./chroma_db"

    
    def __post_init__(self):
        """Configuration validation"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.buffer_size_min >= self.buffer_size_max:
            raise ValueError("Buffer min size must be less than max size")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryConfig':
        """Create configuration from dictionary"""
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False 
