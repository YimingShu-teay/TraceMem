from __future__ import annotations
import argparse
from tracemem.memory.memory import *
from pathlib import Path

def answer_conversation(conv_data):
    memory_system = TraceMem()

    conversation = conv_data.get("conversation", {})
    speaker_a = conversation.get('speaker_a', ' ') 
    speaker_b = conversation.get('speaker_b', ' ')
    
    results = []
    for qa in conv_data.get("qa", []):
        category = qa.get("category", "")
        if category == 5:
            continue
        else:
            question = qa.get("question", "")  
            response =  memory_system.answer(question=question,
                                            speakers=[speaker_a,speaker_b])

            record = {
                "question": question,
                "gt_answer": qa.get("answer"),
                "category": qa.get("category"),
                "evidence": qa.get("evidence", []),
                "tracemem_answer": response,
                }
            results.append(record)

    return results

def fast_answer_dataset(data: List[Dict]) -> None:
    final_results = {}
    for idx, conv_data in enumerate(data):
        results = answer_conversation(conv_data)
        final_results[f"Conversation_{idx}"] = results
        with open('conversation_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
    with open('conversation_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)


def answer_dataset(data: List[Dict]) -> None:
    final_results = {}
    for idx, conv_data in enumerate(data):
        results = answer_conversation(conv_data)
        final_results[f"Conversation_{idx}"] = results
        with open('conversation_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
    with open('conversation_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

# def answer_dataset(data: List[Dict], max_workers: int = 5) -> None:
#     print(f"Starting concurrent processing with {max_workers} workers...")
#     results_dict = {}
    
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_index = {}
#         for i, conv_data in enumerate(data):
#             future = executor.submit(answer_conversation, conv_data)
#             future_to_index[future] = i
        
#         for future in concurrent.futures.as_completed(future_to_index):
#             idx = future_to_index[future]
#             try:
#                 result = future.result()
#                 results_dict[idx] = result
#                 print(f"Processed conversation {idx + 1} ({len(result)} records)")
#             except Exception as e:
#                 print(f"Error processing conversation {idx}: {e}")
#                 results_dict[idx] = []
    
#     final_results = {}
#     for idx in sorted(results_dict.keys()):
#         final_results[f"conversation_{idx+1}"] = results_dict[idx]
    
#     with open('conversation_results.json', 'w', encoding='utf-8') as f:
#         json.dump(final_results, f, ensure_ascii=False, indent=2)
    
#     total_records = sum(len(records) for records in results_dict.values())
#     print(f"All done! Processed {len(data)} conversations, {total_records} records")
#     print(f"Saved to conversation_results.json")

def main() -> None:
    parser = argparse.ArgumentParser(description="Add LoCoMo dataset")
    parser.add_argument("--data_dir", default="dataset/locomo10.json", help="Path to LoCoMo JSON dataset")
    args = parser.parse_args()

    dataset_path = Path(args.data_dir)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    answer_dataset(dataset)

if __name__ == "__main__":
    main()
        


