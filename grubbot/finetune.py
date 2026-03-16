import os
import json
import inspect
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("WARNING: Unsloth not installed. Will fallback to standard transformers if used (not recommended for Grubbot).")

def load_model(model_name: str, max_seq_length: int = 2048):
    if not UNSLOTH_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer
        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer

def formatting_prompts_func(tokenizer):
    def render_chat(conversation, add_generation_prompt: bool = False):
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        rendered = []
        for msg in conversation:
            rendered.append(f"{msg['role'].upper()}: {msg['content']}")
        if add_generation_prompt:
            rendered.append("ASSISTANT:")
        return "\n".join(rendered)

    def wrapper(example):
        texts = []
        for i in range(len(example['messages'])):
            # Extremely simplified text generation - assumes chat template applies 
            # In a real scenario, applying tokenizer.apply_chat_template is preferred
            msgs = example['messages'][i]
            expected_call = example['expected_tool_call'][i]
            
            # Format expected call as the assistant's output
            assistant_response = json.dumps(
                {
                    "name": expected_call["name"],
                    "arguments": expected_call["arguments"],
                }
            )
            
            conversation = [
                {"role": "user", "content": msgs[0]["content"]},
                {"role": "assistant", "content": assistant_response}
            ]
            
            text = render_chat(conversation, add_generation_prompt=False)
            texts.append(text)
            
        return {"text": texts}
    return wrapper

def prepare_dataset(train_path: str, tokenizer):
    dataset = load_dataset("json", data_files=train_path, split="train")
    formatter = formatting_prompts_func(tokenizer)
    dataset = dataset.map(formatter, batched=True)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    keep_cols = [c for c in ["input_ids", "attention_mask"] if c in dataset.column_names]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])
    return dataset

def train(model, tokenizer, dataset, output_dir: str):
    use_cuda = torch.cuda.is_available()

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit" if use_cuda else "adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "data_collator": data_collator,
        "args": training_args,
    }

    params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    return trainer

def save_checkpoint(model, tokenizer, path: str):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
