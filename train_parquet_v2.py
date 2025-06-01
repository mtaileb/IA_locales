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

# Configuration
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
DATA_DIR = "data"
OUTPUT_DIR = "./model"

def load_parquet_datasets(data_dir=DATA_DIR):
    """Charge les datasets train et test depuis les fichiers parquet"""
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
    """Tokenize the dataset"""
    # Combine question and answer
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
    """Charge les données, tokenize et entraîne le modèle"""
    # Chargement des données
    print("Loading datasets...")
    datasets = load_parquet_datasets()
    print(f"Train samples: {len(datasets['train'])}")
    print(f"Test samples: {len(datasets['test'])}")

    # Tokenization
    print("Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=['question', 'answer']
    )

    # Configuration de l'entraînement (version compatible)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_steps=1000,                # Évaluation toutes les 1000 étapes
        save_steps=5000,                # Sauvegarde toutes les 5000 étapes
        save_total_limit=2,              # Max 2 checkpoints
        logging_steps=500,               # Log toutes les 500 étapes
        load_best_model_at_end=True,     # Charge le meilleur modèle à la fin
        fp16=torch.cuda.is_available(),  # Utilise FP16 si GPU disponible
        eval_strategy="steps",     # Stratégie d'évaluation
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Entraînement
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Sauvegarde
    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

def generate_text(prompt, max_length=200):
    """Génère du texte à partir d'un prompt"""
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

if __name__ == '__main__':
    # Initialisation du modèle et tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        model.cuda()

    # Entraînement
    train_model()

    # Exemple de génération
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    print("\nGenerated response:")
    print(generate_text(prompt))
