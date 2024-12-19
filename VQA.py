from flask import Flask, request, render_template, jsonify
from transformers import ViltProcessor
from PIL import Image
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Load model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = torch.load('model/model.pth', map_location=torch.device('cpu'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image and text from request
        image_file = request.files['image']
        question_text = request.form['text']

        # Open image
        image = Image.open(image_file)

        # Prepare inputs
        inputs = processor(image, question_text, return_tensors="pt")
        inputs = {key: value.to('cpu') for key, value in inputs.items()}

        # Forward pass through the model
        outputs = model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        return jsonify({"question": question_text, "answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure the model directory exists
    if not os.path.exists('model/model.pth'):
        raise FileNotFoundError("Model file not found. Ensure 'model/model.pth' exists.")

    app.run(debug=True)
