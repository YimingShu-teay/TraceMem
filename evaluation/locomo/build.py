
from __future__ import annotations
from tracemem.memory.memory import *
import concurrent.futures
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def build_conversation(conv_data: Dict) -> None:

    # memory class
    memory_system = TraceMem()

    # data preperation
    conversation = conv_data['conversation']
    speaker_a = conversation['speaker_a']
    speaker_b = conversation['speaker_b']
    
    # build two cards
    memory_system.build_personal_cards(speaker_a=speaker_a,
                                       speaker_b=speaker_b)
    
    print(f"Completed: {speaker_a} and {speaker_b}")



def build_dataset(data: List[Dict], max_workers: int = 5) -> None:
    print(f"Starting concurrent processing with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        
        for i, conv_data in enumerate(data):
            future = executor.submit(build_conversation, conv_data)
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
    
    build_dataset(dataset, args.max_workers)

if __name__ == "__main__":
    main()

