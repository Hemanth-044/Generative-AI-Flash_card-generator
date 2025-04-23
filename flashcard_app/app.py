from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model locally
model_path = os.path.abspath("./flan-t5-base")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to("cpu")

@app.route('/')
def index():
    return render_template('index.html')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def chunk_text(text, max_chunk_length=1500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_length
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks

def generate_flashcards(text, num_cards=5):
    prompt = f"Generate {num_cards} distinct flashcards (question and answer) from the following notes:\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
    outputs = model.generate(**inputs, max_length=512)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    flashcards = [line.strip() for line in generated.split('\n') if line.strip()]
    return flashcards[:num_cards]

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        num_cards = int(request.form.get('num_cards', 5))
    except:
        num_cards = 5
    num_cards = max(1, min(num_cards, 10))  # clamp between 1 and 10

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    text = extract_text_from_pdf(path)

    chunks = chunk_text(text)
    max_chunks = 10
    flashcards_by_chunk = {}

    for i, chunk in enumerate(chunks[:max_chunks]):
        flashcards = generate_flashcards(chunk, num_cards)
        flashcards_by_chunk[f"Chunk {i+1}"] = flashcards

    return jsonify({'flashcards': flashcards_by_chunk})

if __name__ == '__main__':
    app.run(debug=True)
