from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
 
app = Flask(__name__)
 
# Load the fine-tuned model and tokenizer
model_name = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
 
# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode
 
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
 
    if not prompt:
        return jsonify({'error': 'No prompt provided.'}), 400
 
    # Encode the input and generate the output
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_code = generated_text[len(prompt):].strip()
 
    return jsonify({'code': generated_code})
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# Star API server in command line with: python app.py

# Test API endpoint with: curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Create a simple HTML page with a header"}' http://127.0.0.1:5000/generate
