from typing import List
from .prompts import USER_PROMPT, ANSWER_PROMPT, SEARCH_PROMPT
from ..configs.client import Client
from ..configs.config import MemoryConfig
from ..configs.chroma import ChromaEngine
import json
import os

class AgentReason:
    def __init__(self, llm_client:Client, config:MemoryConfig, backend:ChromaEngine):
        self.llm_client = llm_client
        self.config = config
        self.backend = backend

    def choose_card(self, 
                    question: str,
                    speakers: List[str]):
        
        conv_users = ",".join(speakers)
        input_prompt = f"Question: {question}\nUsers in the conversation: {conv_users}"
        card_choice = self.llm_client.client_response(
            system_prompt=USER_PROMPT,
            input_prompt=input_prompt) 
        
        try:
            user_cards = json.loads(card_choice)
        except json.JSONDecodeError as e:
            print(f"Error {e} in card choice...")

        roles = f"{speakers[0]}_{speakers[1]}"

        choice = user_cards['choice']

        card_paths = []
        for user_card in choice:
            card_path = os.path.join(self.config.cards_dir, f"{roles}_{user_card}.json")
            card_paths.append(card_path)
        
        if not card_paths:
            choice = speakers
            for speaker in speakers:
                card_path = os.path.join(self.config.cards_dir, f"{roles}_{speaker}.json")
                card_paths.append(card_path)
                
        return card_paths, choice
    
    def form_search_prompt(self, 
                           card_paths: List[str],
                           users: List[str]):
        
        print("users=",users)
        print("card_paths=",card_paths)
        contents_prompt = " "
        for idx, card_path in enumerate(card_paths):
            with open(card_path, 'r', encoding='utf-8') as f:
                content = f.read()  
                card_content = json.loads(content)  

            contents_prompt = contents_prompt + f"user name:{users[idx]}\n{card_content}"
        return contents_prompt

    def answer(self,
               question: str,
               speakers: List[str]):
        
        roles = f"{speakers[0]}_{speakers[1]}"
        
        card_paths, users = self.choose_card(question=question, 
                                      speakers=speakers)
        
        contents_prompt = self.form_search_prompt(card_paths=card_paths, users=users)
        
        search_prompt = f"question:{question}\nContents:{contents_prompt}"
        search_results = self.llm_client.client_response(
            system_prompt=SEARCH_PROMPT,
            input_prompt=search_prompt)
        
        try:
            search_results = json.loads(search_results)
        except json.JSONDecodeError as e:
            print(f"Error {e} in search...")
        
        search_threads = search_results["results"]
 
        episode_results = self.backend.search_episodes(user_id=roles,
                                                        query=question,
                                                        top_k=20)  

        episode_contents = [search_result['content'] for search_result in episode_results]   
        episode_prompt = "\n".join(episode_contents)

        answer_prompt = f"Question: {question}\n\nContents:\nEpisodes:\n{episode_prompt}\n\n"
        
        for idx, search_thread in enumerate(search_threads):
        
            user = list(search_thread.keys())[0]
            search_thread = search_thread[user]
            if search_thread:
                thread_ids = list(set(item["thread_id"] for item in search_thread))
                
                thread_collection = self.backend._get_thread_collection(user_id=f"{roles}_{user}")
                threads = thread_collection.get(ids=thread_ids, include=['documents', 'metadatas'])
                threads = threads['documents']
                threads = "\n".join(threads)  
                answer_prompt = answer_prompt + f"{user} threads:\n" + threads

        response = self.llm_client.client_response(
            system_prompt=ANSWER_PROMPT,
            input_prompt=answer_prompt)

        return response
