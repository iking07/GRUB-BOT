import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from .lora import patch_model_with_lora

class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                example = json.loads(line)
                
                # Assume conversational format
                prompt = ""
                for msg in example.get("messages", []):
                    prompt += f"{msg['role'].upper()}: {msg['content']}\n"
                
                # Format expected call as the assistant's output
                expected_call = example.get("expected_tool_call", {})
                if expected_call:
                    prompt += "ASSISTANT: "
                    target = json.dumps({
                        "name": expected_call.get("name"),
                        "arguments": expected_call.get("arguments")
                    })
                    self.samples.append((prompt, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, target = self.samples[idx]
        full_text = prompt + target + self.tokenizer.eos_token
        
        # Tokenize prompt and target separately to mask prompt loss
        tokenized_prompt = self.tokenizer(prompt, add_special_tokens=False)
        tokenized_full = self.tokenizer(
            full_text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokenized_full["input_ids"][0]
        attention_mask = tokenized_full["attention_mask"][0]
        
        labels = input_ids.clone()
        prompt_len = len(tokenized_prompt["input_ids"])
        labels[:prompt_len] = -100  # Ignore index for CrossEntropyLoss
        labels[attention_mask == 0] = -100 # Ignore padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train_custom(model_name: str, train_path: str, output_dir: str, epochs=3, batch_size=4, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading bare model {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Patch with custom LoRA from scratch
    print("Patching linear layers with custom LoRA...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    model = patch_model_with_lora(model, target_modules, r=16, lora_alpha=32)
    model.to(device)
    
    # Verify trainability
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} || All params: {all_params} || Trainable %: {100 * trainable_params / all_params:.2f}%")
    
    dataset = InstructionDataset(train_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training finished natively.")
    return model, tokenizer

if __name__ == "__main__":
    train_custom("gpt2", "data/train.jsonl", "models/custom_lora_run")