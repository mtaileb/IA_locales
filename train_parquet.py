import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Load separate train and test datasets
dataset = load_dataset(
    'json',
    data_files={
        'train': '/content/data/train.jsonl',
        'test': '/content/data/test.jsonl'
    }
)

# Tokenize the dataset
def tokenize_function(example):
    text = example['question'] + "\n" + example['answer']
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Apply tokenization to train and test splits
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=['question', 'answer']
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=5000,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

# Text generation function
def generate_code(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example
prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
print(generate_code(prompt))
