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
model_name = "EleutherAI/gpt-neo-1.3B"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add this line to set a padding token (using EOS token as default)
tokenizer.pad_token = tokenizer.eos_token  # <-- Fix here

model = AutoModelForCausalLM.from_pretrained(model_name)
 
# Load your custom dataset
dataset = load_dataset('json', data_files={'train': '/content/data/data.jsonl'})
 
# Tokenize the dataset
'''
def tokenize_function(example):
    tokens = tokenizer(
        example['prompt'] + "\n" + example['code'],
        truncation=True,
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
'''

def tokenize_function(example):
    # Combine prompt + code
    text = example['prompt'] + "\n" + example['code']
    
    # Tokenize with padding AND truncation
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",  # <-- Critical addition
        max_length=512,       # Same as before
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

 
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=['prompt', 'code']
)


# Split the dataset into training and validation sets
tokenized_datasets = tokenized_datasets['train'].train_test_split(test_size=0.1)

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
    fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU is available
)

# Data collator for language modeling
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


# Start model finetuning
trainer.train()

# Save the finetuned model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

# Generate code samples
def generate_code(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)
 
# Example
prompt = "Créer une fonction Python qui additionne trois nombres."
print(generate_code(prompt))

# Evaluate the Output
#    Syntax Checks: Use linters like ESLint.
#    Functional Tests: Run the code in a development environment.
#    Peer Review: Have others review the code for quality.
