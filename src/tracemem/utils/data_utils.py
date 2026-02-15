import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List
import chromadb
import numpy as np
import re
import uuid
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def safe_extract_text(root, tag_name):
    element = root.find(tag_name)
    if element is not None and element.text is not None:
        return element.text.strip()
    return ""

def parse_segement_result(xml_string, buffer_len, buffer_max):
    try:
        cleaned_string = xml_string.strip()
        
        if '```' in cleaned_string:
            match = re.search(r'<analysis[^>]*>.*?</analysis>', cleaned_string, re.DOTALL)
            if match:
                cleaned_string = match.group(0)
            else:
                cleaned_string = re.sub(r'```[a-zA-Z]*\n?|```', '', cleaned_string).strip()
        
        cleaned_string = re.sub(r'&(?!(amp|lt|gt|apos|quot);)', '&amp;', cleaned_string)

        root = ET.fromstring(cleaned_string)
        
        text_number = root.get('text_number', None)
        
        result = {
            'text_number': int(text_number) if text_number else None,
            'keywords': safe_extract_text(root, 'keywords'),
            'topic_shift_label': safe_extract_text(root, 'topic_shift_label') == 'YES' if buffer_len < buffer_max else True,
            'current_summary': safe_extract_text(root, 'current_summary'),
            'semantic_memory': safe_extract_text(root, 'semantic_memory'),
        }   
        
        result['topic_shift_reason'] = safe_extract_text(root, 'topic_shift_reason')
        
        if result['keywords']:
            result['keywords_list'] = [k.strip() for k in result['keywords'].split(',')]
        else:
            result['keywords_list'] = []

        if result['semantic_memory']:
            memory_items = [item.strip() for item in result['semantic_memory'].split(';') if item.strip()]
            result['semantic_memory_list'] = memory_items
        else:
            result['semantic_memory_list'] = []
            
        if result['episode_title'] == '':
            result['episode_title'] = None
            
        return result
        
    except ET.ParseError as e:
        print(f"XML Error: {e}")
        print(f"Cleaned string attempt: {cleaned_string[:200]}...") 
        return None
    except Exception as e:
        print(f"An error occurred during the parsing process: {e}")
        return None

def check_chroma_collections(db_path:str):
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    if not collections:
        return
    print(f"{'Collection Name':<30} | {'Count':<10}")
    print("-" * 45)
    for col in collections:
        count = col.count()
        print(f"{col.name:<30} | {count:<10}")

    return collections


def get_episodes_from_collection(db_path:str,collection_name:str):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    results = collection.get(include=['embeddings', 'documents', 'metadatas'])
    return results


def parse_timestamp(value: str) -> datetime:
    """Parse dataset timestamps such as "1:56 pm on 8 May, 2023"."""
    value = " ".join(value.split())
    if " on " not in value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now()
    time_part, date_part = value.split(" on ")
    time_part = time_part.lower().strip()
    hour = 0
    minute = 0
    is_pm = "pm" in time_part
    time_part = time_part.replace("pm", "").replace("am", "").strip()
    if ":" in time_part:
        hour_str, minute_str = time_part.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    else:
        hour = int(time_part)
    if is_pm and hour != 12:
        hour += 12
    if not is_pm and hour == 12:
        hour = 0
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    parts = date_part.replace(",", "").split()
    day = 1
    month = 1
    year = datetime.now().year
    for part in parts:
        lower = part.lower()
        if lower in months:
            month = months[lower]
        elif part.isdigit():
            num = int(part)
            if num > 31:
                year = num
            else:
                day = num
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute)


def build_sessions_locomo(conversation: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")
    special_keys = {"speaker_a", "speaker_b"}
    
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    
    for key, chats in conversation.items():
        if key in special_keys or key.endswith("_date_time"):
            continue
        
        if key.startswith("session_"):
            session_name = key
            timestamp_raw = conversation.get(f"{key}_date_time")
            timestamp = parse_timestamp(timestamp_raw) if timestamp_raw else datetime.now()
            
            session_messages: List[Dict[str, Any]] = []
            
            for chat in chats or []:
                speaker = chat.get("speaker", speaker_a)
                text = chat.get("text", "")
                
                parts = [text]
                if chat.get("blip_caption"):
                    parts.append(f"[Image: {chat['blip_caption']}]")
                if chat.get("query"):
                    parts.append(f"[Search: {chat['query']}]")
                
                role = speaker if speaker in (speaker_a, speaker_b) else "user"
                
                session_messages.append({
                    "role": role,
                    "content": " ".join(parts),
                    "timestamp": timestamp.isoformat(),
                    "metadata": {
                        "original_speaker": speaker,
                        "dataset_timestamp": timestamp_raw,
                        "blip_caption": chat.get("blip_caption"),
                        "search_query": chat.get("query"),
                        "session": session_name,  
                        "dia_id": chat.get("dia_id"),  
                    },
                })
            
            sessions[session_name] = session_messages
    
    return sessions

def parse_output_to_list(output_text):
    pattern = r'OUTPUT_MEMORY_LIST:\s*\[\s*(.*?)\s*\]'
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        lines = output_text.strip().split('\n')
        memory_list = []
        in_memory_list = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('OUTPUT_MEMORY_LIST:'):
                in_memory_list = True
                continue
            if in_memory_list and line.startswith('['):
                continue
            if in_memory_list and line.startswith(']'):
                break
            if in_memory_list and line:
                line = line.strip(',')
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                elif line.startswith("'") and line.endswith("'"):
                    line = line[1:-1]
                memory_list.append(line)
        
        return [item.strip() for item in memory_list if item.strip()]
    
    content = match.group(1)

    items = []

    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line == ',':
            continue
        
        line = line.strip(',')
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        elif line.startswith("'") and line.endswith("'"):
            line = line[1:-1]
        
        items.append(line)

    return [item for item in items if item]


def plot_memory_landscape(ax, embeddings_2d, labels,user_id,level,n_samples):

    x_range = embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()
    y_range = embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min()
    
    pad_ratio = 0.4 
    x_min = embeddings_2d[:, 0].min() - x_range * pad_ratio
    x_max = embeddings_2d[:, 0].max() + x_range * pad_ratio
    y_min = embeddings_2d[:, 1].min() - y_range * pad_ratio
    y_max = embeddings_2d[:, 1].max() + y_range * pad_ratio
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                        np.linspace(y_min, y_max, 250))
    
    
    if len(embeddings_2d) >= 3:
        try:
            kde = gaussian_kde(embeddings_2d.T, bw_method=0.3)
            zz = np.reshape(kde(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
            
            cntr = ax.contourf(xx, yy, zz, levels=30, cmap='PuBu', alpha=0.5, zorder=1)
            ax.contour(xx, yy, zz, levels=5, colors='white', alpha=0.2, linewidths=0.5, zorder=2)
            
            cbar = plt.colorbar(cntr, ax=ax, shrink=0.7, pad=0.05)
            cbar.ax.tick_params(labelsize=7)
        except: pass
    
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='white', alpha=0.6, s=15, edgecolors='none', zorder=4)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_title(f'{user_id} {level} Clustering ({n_samples} data)', fontsize=18, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    random_id = uuid.uuid4().hex[:8]  
    filename = f"memory_landscape_{user_id}_{random_id}.png"
    
    plt.savefig(filename, bbox_inches='tight', dpi=600, facecolor='white')
    print(f"Plot saved as: {filename}")

def visualize_clusters(reduced_embeddings, labels, data,user_id,level):
    try:
        n_samples = len(reduced_embeddings)
        
        embeddings_2d = reduced_embeddings
        
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111) 

        plot_memory_landscape(ax, embeddings_2d, labels,user_id,level,n_samples)
        
        plt.tight_layout()
                    
    except Exception as e:
        print(f"{e}")
        import traceback
        traceback.print_exc()

