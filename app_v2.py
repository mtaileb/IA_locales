from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
import time
from threading import Thread

app = Flask(__name__)

# Load model and tokenizer (your existing code)
model_name = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')  # Fixed typo: 'prompt' instead of 'prompt'
    
    if not prompt:
        return jsonify({'error': 'No prompt provided.'}), 400

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

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False)  # debug=False avoids signal error

if __name__ == '__main__':
    # Start server in a thread
    server_thread = Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # Wait for server to start (3 seconds is usually enough)
    time.sleep(3)

    # Test the API
    test_prompt = "How many centiliters is there is half a liter?"
    try:
        response = requests.post(
            "http://127.0.0.1:5000/generate",
            json={"prompt": test_prompt},  # Fixed typo: 'prompt' instead of 'prompt'
            headers={"Content-Type": "application/json"}
        )
        print("API Response:", response.json())
    except Exception as e:
        print("Test failed:", str(e))

    # Keep the main thread alive (optional)
    input("Press Enter to stop the server...")
