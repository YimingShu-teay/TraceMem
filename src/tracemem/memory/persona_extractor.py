from .prompts import PERSONA_MODEL_PROMPT
from ..configs.client import Client
from typing import Dict, List
import json

class PersonaExtractor:
    def __init__(self, llm_client: Client):
        self.llm_client = llm_client
    
    def format_experience_prompt(self, 
                                 topic: Dict,
                                 speaker: str) -> str:
        episode_summary = topic['summary']
        labeled_memories = []

        memory_texts = topic['semantic_memories'].get(speaker, [])
        for memory_text in memory_texts:
            labeled_memories.append(f"{speaker}: {memory_text}")
        
        formatted_input = f"""Speaker: {speaker}
        **Episode Summary**: {episode_summary}
        **Labeled Semantic Memories**:{chr(10).join([' - ' + mem for mem in labeled_memories])}"""
        
        return formatted_input
    
    def format_gap_experience_prompt(self, 
                                     topic: Dict,
                                     user: str) -> str:
        episode_summary = topic['summary']
        labeled_memories = []
        for memory in topic['semantic_memories']:
            memory_text, speaker = memory
            if speaker == user:
                labeled_memories.append(f"{speaker}: {memory_text}")
        
        formatted_input = f"""Episode Memory: {episode_summary}\nSemantics Memories: {chr(10).join(['- ' + mem for mem in labeled_memories])}"""
        
        return formatted_input
        
    def experiences_extraction(self, topics: List[Dict]) -> List[Dict]:
        for topic in topics:
            speakers = topic['semantic_memories'].keys()
            topic['experience'] = {}
            
            for speaker in speakers:
                input_prompt = self.format_experience_prompt(topic,speaker)       
                experience = self.llm_client.client_response(system_prompt=PERSONA_MODEL_PROMPT, input_prompt=input_prompt)
                experience = json.loads(experience)
                topic['experience'][speaker] = experience

        return topics