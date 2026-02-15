import json
import pickle
import uuid
from typing import Any, Union, Dict

import logging
logger = logging.getLogger(__name__)

class RedisDataProcessor:
    
    PICKLE_MARKER = b"__PICKLE__"
    UUID_LENGTH = 8

    @staticmethod
    def process_data_for_storage(data: Any) -> Union[str, bytes]:
        if isinstance(data, str):
            serialized = data
        else:
            try:
                serialized = json.dumps(data, ensure_ascii=False)
            except:
                serialized = RedisDataProcessor.PICKLE_MARKER + pickle.dumps(data)
        
        unique_id = str(uuid.uuid4())[:8]
        if isinstance(serialized, bytes):
            return unique_id.encode('utf-8') + b":" + serialized
        return f"{unique_id}:{serialized}"

    @staticmethod
    def process_data_from_storage(member: Union[str, bytes]) -> Dict[str, Any]:
        if isinstance(member, bytes):
            unique_id_bytes, raw_data = member.split(b":", 1)
            if raw_data.startswith(RedisDataProcessor.PICKLE_MARKER):
                data = pickle.loads(raw_data[len(RedisDataProcessor.PICKLE_MARKER):])
            else:
                try: data = json.loads(raw_data.decode('utf-8'))
                except: data = raw_data.decode('utf-8')
        else:
            unique_id, raw_data = member.split(':', 1)
            try: data = json.loads(raw_data)
            except: data = raw_data
            
        return {"id": unique_id, "data": data}