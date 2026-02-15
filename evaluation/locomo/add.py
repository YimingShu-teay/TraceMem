from __future__ import annotations
import argparse
import json
from tracemem.memory.memory import *
import concurrent.futures
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tracemem.utils.data_utils import build_sessions_locomo

def add_conversation(conv_data: Dict) -> None:

    # memory class
    memory_system = TraceMem()

    # data preperation
    conversation = conv_data['conversation']
    sessions = build_sessions_locomo(conversation)
    speaker_a = conversation['speaker_a']
    speaker_b = conversation['speaker_b']
    roles = f"{speaker_a}_{speaker_b}"
    
    # add sessions
    memory_system.add_memories(sessions=sessions,
                               roles=roles)
    
    print(f"Completed: {speaker_a} and {speaker_b}")


def add_dataset(data: List[Dict], max_workers: int = 5) -> None:
    print(f"Starting concurrent processing with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        
        for i, conv_data in enumerate(data):
            future = executor.submit(add_conversation, conv_data)
            future_to_index[future] = i + 1
        
        completed_count = 0
        failed_count = 0
        
        for future in concurrent.futures.as_completed(future_to_index):
            conv_num = future_to_index[future]            
            try:
                future.result()
                completed_count += 1
                print(f"Conversation {conv_num}/10 processed successfully")                
            except Exception as e:
                failed_count += 1
                print(f"Conversation {conv_num}/10 failed: {type(e).__name__}: {e}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Add LoCoMo dataset into Nemori memory")
    parser.add_argument("--data_dir", default="dataset/locomo10.json", help="Path to LoCoMo JSON dataset")
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    dataset_path = Path(args.data_dir)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    
    add_dataset(dataset, args.max_workers)

if __name__ == "__main__":
    main()


