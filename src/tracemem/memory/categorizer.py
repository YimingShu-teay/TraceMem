import json
import umap
import numpy as np
from typing import Dict, List
import hdbscan
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from ..configs.config import MemoryConfig
from ..configs.chroma import ChromaEngine
from ..configs.client import Client
from ..storage.thread import ThreadMemory
from .prompts import TOPIC_PROMPT, THREAD_PROMPT, THEME_PROMPT

class Categorizer:
    def __init__(self, backend: ChromaEngine, config: MemoryConfig, llm_client: Client):
        self.backend = backend
        self.config = config
        self.llm_client= llm_client
        self.min_samples = 1
        self.thread_map = {}
    
    def _fetch_data(self, user_id: str):
        collection = self.backend._get_experience_collection(user_id=user_id)
        results = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        if results['embeddings'] is None or len(results['embeddings']) == 0:
            raise ValueError(f"Collection '{user_id}' has no data.")
        
        # Print loading statistics
        print(f"Loaded {len(results['embeddings'])} memories")
        print(f"Vector dimension: {len(results['embeddings'][0])}")
        
        # Calculate and display vector norm statistics
        embeddings = np.array(results['embeddings'])
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Vector norm statistics - Mean: {np.mean(norms):.3f}, Std: {np.std(norms):.3f}")
        print(f"Vector norm range: [{np.min(norms):.3f}, {np.max(norms):.3f}]")
        
        # Return all fetched data
        return {
            "embeddings": embeddings,
            "metadatas": results['metadatas'],
            "documents": results['documents'],
            "ids": results['ids']}
    
    def run_clustering(self, 
                       data: Dict, 
                       n_neighbors: int,
                       min_cluster_size: int,
                       use_pca=True):

        embeddings = np.array(data["embeddings"])
        
        if use_pca:
            n_components = min(50, len(embeddings) - 1, embeddings.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            features = pca.fit_transform(embeddings)
        else:
            features = embeddings
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            n_components=2, 
            min_dist=0.01,
            metric='cosine',
            random_state=None)
        
        reduced_embeddings = reducer.fit_transform(features)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True)
        
        labels = clusterer.fit_predict(reduced_embeddings)

        noise_mask = (labels == -1)
        if np.any(noise_mask):
            print(f"Detect {np.sum(noise_mask)} noise, classifying...")
            soft_scores = hdbscan.all_points_membership_vectors(clusterer)
            
            for i in np.where(noise_mask)[0]:
                if np.all(soft_scores[i] <= 0):
                    non_noise_mask = ~noise_mask
                    if np.any(non_noise_mask):
                        knn = KNeighborsClassifier(n_neighbors=min(5, np.sum(non_noise_mask)))
                        knn.fit(reduced_embeddings[non_noise_mask], labels[non_noise_mask])
                        labels[i] = knn.predict(reduced_embeddings[i].reshape(1, -1))[0]
                else:
                    labels[i] = np.argmax(soft_scores[i])
        
        return self._format_results(data, labels, clusterer)

    def _format_results(self, data, labels, clusterer=None):
        clusters = {}
        
        for idx, label in enumerate(labels):
            label_key = int(label)
            if label_key not in clusters:
                clusters[label_key] = {
                    "ids": [],
                    "embeddings": [],
                    "documents": [],
                    "metadatas": [],
                    "probabilities": [] if clusterer and hasattr(clusterer, 'probabilities_') else None}

            clusters[label_key]["ids"].append(data["ids"][idx])
            clusters[label_key]["embeddings"].append(data["embeddings"][idx])
            clusters[label_key]["documents"].append(data["documents"][idx])
            clusters[label_key]["metadatas"].append(data["metadatas"][idx])
            
            if clusterer and hasattr(clusterer, 'probabilities_'):
                clusters[label_key]["probabilities"].append(clusterer.probabilities_[idx])
        
        sorted_clusters = {}
        sorted_labels = sorted(
            clusters.keys(), 
            key=lambda x: (x == -1, -len(clusters[x]["ids"]) if x != -1 else 0))
        
        for label in sorted_labels:
            sorted_clusters[label] = clusters[label]
        
        return sorted_clusters


    
    def topic_categorize(self,
                         roles: str, 
                         user_id: str):
        
        data = self._fetch_data(user_id=f"{roles}_{user_id}")
        topic_clusters = self.run_clustering(data=data,
                                      n_neighbors=10,
                                      min_cluster_size=5)
        topics = {}
        topics['topics'] = []
        for _, topic_cluster in topic_clusters.items():
            input_prompt = "\n".join(topic_cluster['documents'])
            
            topics_result = self.llm_client.client_response(
                system_prompt=TOPIC_PROMPT,
                input_prompt=input_prompt)  
            
            try:
                topic = json.loads(topics_result)
            except json.JSONDecodeError as e:
                print(f"Error {e} in topic generation...")

            topics['topics'].append(topic)
        
        topics_input = json.dumps(topics, ensure_ascii=False, indent=2)
        themes = self.llm_client.client_response(
            system_prompt=THEME_PROMPT,
            input_prompt=topics_input) 
        
        try:
            themes = json.loads(themes)   
        except json.JSONDecodeError as e:
            print(f"Error {e} in topic summary generation...")
                      
        themes.update(topics)
        return topic_clusters, themes
    
    def thread_categorize(self, 
                          topic_clusters: Dict[str, Dict], 
                          topics: Dict):
        for topic_idx, (_, topic_cluster) in enumerate(topic_clusters.items()):
            # thread clustering for one topic
            thread_clusters = self.run_clustering(data=topic_cluster,
                                                  n_neighbors=2,
                                                  min_cluster_size=2)
            
            threads = []   
            for _, thread_cluster in thread_clusters.items():       
                # summarize experiences and formulate the one thread
                thread = self.experience_summarize(cluster=thread_cluster) # exper_summaries: List
                threads.append(thread)
            topics["topics"][topic_idx]['threads'] = threads

        return topics
    
    def experience_summarize(self, cluster: Dict[str, Dict]) -> List:

        thread_id = self.add_thread_memory(cluster)

        exper_contents = cluster["documents"]

        contents_prompt = "\n".join(exper_contents)
        thread_summary = self.llm_client.client_response(
            system_prompt=THREAD_PROMPT,
            input_prompt=contents_prompt) 
        
        try:
            thread_summary = json.loads(thread_summary)
        except json.JSONDecodeError as e:
            print(f"Error {e} in thread summary generation...")
        
        thread_summary.update({"thread_id":thread_id})
        return thread_summary
                     

    def add_thread_memory(self, cluster: Dict[str, Dict]):
        
        documents = cluster["documents"]
        episode_ids = [metadata["source_episode"] for metadata in cluster["metadatas"]]
        user_id = cluster["metadatas"][0]['user_id']

        thread_content = " ".join(documents)
        
        thread_memory = ThreadMemory(
            content=thread_content,
            source_episode=json.dumps(episode_ids),
            user_id=user_id)

        self.backend.add_thread_memory(user_id=user_id, thread=thread_memory)
        return thread_memory.thread_id
