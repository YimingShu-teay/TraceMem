import re
from ..configs.client import Client
from .prompts import SEGMENT_PROMPT
from typing import List

class TopicSegmentor:
    def __init__(self,llm_client:Client):
        self.llm_client = llm_client
        
    def format_segment_prompt(self, messages_seg: List[str]):
        if not messages_seg:
            return "", []
        
        first_msg_metadata = messages_seg[0].get('metadata', {})
        msgs_time = first_msg_metadata.get('dataset_timestamp', 'Unknown Time')
        
        formatted_lines = []
        speakers = []
        formatted_lines.append(f"Current Time: {msgs_time}")
        
        for i, msg in enumerate(messages_seg):
            speaker = msg.get('role', 'Unknown')
            speakers.append(speaker)
            
            content = msg.get('content', '')
            text_only = re.sub(r'\[.*?\]', '', content).strip()
            
            metadata = msg.get('metadata', {})
            blip = metadata.get('blip_caption')
            # search_query = metadata.get('search_query') or metadata.get('query')
            
            display_parts = []
            if text_only:
                display_parts.append(text_only)
            if blip:
                display_parts.append(f"Image: {blip}")
            # if search_query:
            #     display_parts.append(f"[Search: {search_query}]")
                
            if display_parts:
                final_content = " ".join(display_parts)
                formatted_line = f"<D{i+1}>{speaker}: {final_content}</D{i+1}>"
                formatted_lines.append(formatted_line)
                
        return "\n".join(formatted_lines), speakers
    
    def extract_topics(self, 
                       result_text: str, 
                       speakers: str):
        pattern = r'<D(\d+)>.*?<intent>(.*?)</intent>.*?<semantic>(.*?)</semantic>.*?</D\1>'
        alt_pattern = r'<D(\d+)>.*?<intent>(.*?)</intent>.*?<semantic>(.*?)</semantic>.*?</D\1>'
        alt_pattern_no_intent = r'<D(\d+)>.*?<semantic>(.*?)</semantic>.*?</D\1>'
        
        matches = re.findall(pattern, result_text, re.DOTALL)
        if not matches:
            matches = re.findall(alt_pattern, result_text, re.DOTALL)
            matches = [(idx, intent, semantic.strip(), "") for idx, intent, semantic in matches]
        
        if not matches:
            matches = re.findall(alt_pattern_no_intent, result_text, re.DOTALL)
            matches = [(idx, "DEVELOP_TOPIC", semantic.strip(), "") for idx, semantic in matches]
        
        matches = sorted([(int(idx), intent, semantic.strip()) 
                        for idx, intent, semantic in matches], 
                        key=lambda x: x[0])
        
        topics = []
        current_topic = []
        current_start_idx = None  
        
        for idx, intent, semantic in matches:
            idx_0based = idx - 1
            
            if intent == "CHANGE_TOPIC" and current_topic:
                semantic_dict = {}
                for memory, speaker in current_topic:
                    if speaker not in semantic_dict:
                        semantic_dict[speaker] = []
                    semantic_dict[speaker].append(memory)
                
                topics.append({
                    'range': (current_start_idx, idx_0based - 1),  
                    'semantic_memories': semantic_dict, 
                    'count': len(current_topic)
                })
                current_topic = []
                current_start_idx = idx_0based

            if current_start_idx is None:
                current_start_idx = idx_0based
            
            speaker = speakers[idx_0based] if 0 <= idx_0based < len(speakers) else "Unknown"

            if semantic:
                current_topic.append((semantic, speaker))
        
        if current_topic and current_start_idx is not None:
            semantic_dict = {}
            for memory, speaker in current_topic:
                if speaker not in semantic_dict:
                    semantic_dict[speaker] = []
                semantic_dict[speaker].append(memory)
            
            topics.append({
                'range': (current_start_idx, matches[-1][0] - 1), 
                'semantic_memories': semantic_dict,  
                'count': len(current_topic)
            })
        return topics

    def topic_segment_session(self, messages: List[str]):
        input_prompt,speakers = self.format_segment_prompt(messages)
        result = self.llm_client.client_response(system_prompt=SEGMENT_PROMPT,
                                                 input_prompt=input_prompt)
        topics = self.extract_topics(result,speakers)        
        return topics
    