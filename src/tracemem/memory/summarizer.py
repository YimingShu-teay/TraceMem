from ..configs.client import Client
from typing import Dict, List
from .prompts import EPISODE_PROMPT

class Summarizer:
    def __init__(self, llm_client: Client):
        self.llm_client = llm_client

    def format_episode_prompt(self,dialogue_data):
        if not dialogue_data:
            return "Current Time: Unknown\nNo dialogue content."
        
        cleaned_data = []
        for entry in dialogue_data:
            cleaned_entry = {
                'role': entry.get('role', 'Unknown'),
                'content': entry.get('content', ''),
                'timestamp': entry.get('metadata',"").get('dataset_timestamp', '')
            }
            
            if cleaned_entry['role'].startswith('olved'):
                cleaned_entry['role'] = 'Maria'
            
            cleaned_data.append(cleaned_entry)
        
        current_time = next((entry['timestamp'] for entry in cleaned_data if entry['timestamp']), "Unknown")        
        lines = [f"Current Time: {current_time}"]
        
        for entry in cleaned_data:
            role = entry['role']
            content = entry['content']
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def episodes_summary(self, topics:List[Dict], mesaages:List[str]) -> Dict:
        for topic in topics:
            sidx,eidx = topic['range'][0], topic['range'][1]
            msgs = mesaages[sidx:eidx+1]
            input_prompt = self.format_episode_prompt(msgs)            
            summary = self.llm_client.client_response(system_prompt=EPISODE_PROMPT, input_prompt=input_prompt)
            topic['summary'] = summary
        return topics
    
