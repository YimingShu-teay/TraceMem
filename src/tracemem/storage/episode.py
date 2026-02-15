from dataclasses import dataclass, field
from typing import List
from datetime import datetime
import uuid


@dataclass
class Episode:
    # title: str                                      
    summary: str  
    # keywords: str                                                                                          
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))  
    user_id: str  = ""                         
    created_at: datetime = field(default_factory=datetime.now)         
    timestamp: datetime = field(default_factory=datetime.now)          
    tags: List[str] = field(default_factory=list)          
    
    def __str__(self) -> str:
        return f"Episode(id={self.episode_id}"
    
    def __repr__(self) -> str:
        return self.__str__() 