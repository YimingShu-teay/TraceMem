from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class ThreadMemory:
    content: str            
    # keywords: str                                                             
    user_id: str     
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))                                    
    created_at: datetime = field(default_factory=datetime.now)         
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))  
    
    source_episode: str = field(default_factory=list) 
    
    updated_at: Optional[datetime] = None           

    def __str__(self) -> str:
        return f"Thread(id={self.memory_id}"
    
    def __repr__(self) -> str:
        return self.__str__() 