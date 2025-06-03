import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def load_model(model_dir):
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Set padding token (required for some generation configs)
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a finetuned GPT-Neo model.")
    parser.add_argument("--model_dir", type=str, default="./model", help="Path to the finetuned model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to generate from")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum generation length")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)
    output = generate_text(args.prompt, model, tokenizer, max_length=args.max_length)

    print("\nGenerated Output:\n")
    print(output)

# Exemple: python run_inference.py --prompt "How many centiliters is there is half a liter?"

