import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import os
import pandas as pd
from datasets import Dataset, DatasetDict
import argparse
import wandb

# Configuration
BASE_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
DATA_DIR = "data"
OUTPUT_DIR = "./model"

# Désactive wandb par défaut
os.environ["WANDB_DISABLED"] = "true"

def load_parquet_datasets(data_dir=DATA_DIR):
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

def tokenize_function(examples):
    texts = [q + "\n" + a for q, a in zip(examples['question'], examples['answer'])]
    tokens = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

def train_model():
    print("Loading datasets...")
    datasets = load_parquet_datasets()
    print(f"Train samples: {len(datasets['train'])}")
    print(f"Test samples: {len(datasets['test'])}")

    print("Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=['question', 'answer']
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_steps=1000,
        save_steps=5000,
        save_total_limit=2,
        logging_steps=500,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=finetuned_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    finetuned_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

def generate_text(prompt, model, tokenizer, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def compare_models(prompt):
    print("\n--- Base Model ---")
    base_output = generate_text(prompt, base_model, base_tokenizer)
    print(base_output)

    print("\n--- Finetuned Model ---")
    finetuned_output = generate_text(prompt, finetuned_model, tokenizer)
    print(finetuned_output)

if __name__ == '__main__':
    print("Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    if torch.cuda.is_available():
        base_model.cuda()

    print("Loading finetuned model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(OUTPUT_DIR):
        finetuned_model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    else:
        finetuned_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

    if torch.cuda.is_available():
        finetuned_model.cuda()

    # Train (si besoin)
    train_model()

    # Comparaison de génération
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    compare_models(prompt)
