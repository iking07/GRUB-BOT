import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from grubbot.eval import FailedExample
from grubbot.cluster import embed_failures, cluster_failures

def evaluate_model(model, tokenizer, test_file="data/eval.jsonl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    total = 0
    correct_tools = 0
    valid_json = 0
    correct_params = 0
    
    failures = []
    
    with open(test_file, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Evaluating"):
            if not line.strip(): continue
            example = json.loads(line)
            
            user_query = ""
            prompt = ""
            for msg in example.get("messages", []):
                prompt += f"{msg['role'].upper()}: {msg['content']}\n"
                if msg['role'] == 'user':
                    user_query = msg['content']
                    
            prompt += "ASSISTANT: "
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.01
                )
            
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # evaluate
            expected_call = example.get("expected_tool_call", {})
            total += 1
            
            is_valid_json = False
            is_correct_tool = False
            is_exact_match = False
            error_type = "invalid_json"
            
            try:
                pred = json.loads(generated.strip())
                is_valid_json = True
                valid_json += 1
                error_type = "wrong_tool"
                
                if pred.get("name") == expected_call.get("name"):
                    is_correct_tool = True
                    correct_tools += 1
                    error_type = "wrong_parameters"
                    
                    if pred.get("arguments") == expected_call.get("arguments"):
                        is_exact_match = True
                        correct_params += 1
                        error_type = "none"
            except:
                pass
                
            if not is_exact_match:
                failures.append(FailedExample(
                    expected=expected_call,
                    predicted=generated.strip(),
                    error_type=error_type,
                    user_query=user_query
                ))
                
    print(f"Total Evaluated: {total}")
    print(f"Valid JSON: {valid_json}/{total} ({valid_json/total*100:.2f}%)")
    print(f"Correct Tool: {correct_tools}/{total} ({correct_tools/total*100:.2f}%)")
    print(f"Exact Match (Params): {correct_params}/{total} ({correct_params/total*100:.2f}%)")
    
    if failures:
        print(f"\nClustering {len(failures)} failures...")
        try:
            embeddings = embed_failures(failures)
            clusters = cluster_failures(failures, embeddings)
            for c in clusters:
                print(f"Cluster [{c.label}] (Size: {c.size})")
                for ex in c.examples[:2]:  # Show top 2 from each cluster
                    print(f"  - Query: {ex.user_query[:50]}... | Pred: {ex.predicted}")
        except Exception as e:
            print(f"Clustering failed (is sentence-transformers installed?): {e}")

if __name__ == "__main__":
    from grubbot.custom_train import patch_model_with_lora
    # For testing the evaluation loop script:
    model_name = "models/custom_lora_run"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        evaluate_model(model, tokenizer)
    except Exception as e:
        print(f"Could not load model to evaluate: {e}")
