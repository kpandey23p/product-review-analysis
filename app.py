from flask import Flask, request, jsonify, render_template
import torch
import json
import pandas as pd
import gzip
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
app = Flask(__name__)

# Load dataset & vocab
df = pd.read_json("dataset.json.gz", lines=True)
with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Load trained model
class CapsuleBiLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.bilstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return self.sigmoid(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CapsuleBiLSTM(len(vocab), 128, 256, 1).to(device)
model.load_state_dict(torch.load("capsule_bilstm_model.pth", map_location=device))
model.eval()

# Fetch reviews
def fetch_reviews(product_id):
    return df[df['asin'] == product_id]['text'].dropna().tolist()

# Predict
def predict_product(product_id):
    reviews = fetch_reviews(product_id)
    if not reviews:
        return {"message": "No reviews found for this product."}

    scores = []
    for review in reviews:
        tokens = word_tokenize(review.lower())
        encoded_review = [vocab.get(word, 1) for word in tokens]
        encoded_review = torch.tensor(encoded_review, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            score = model(encoded_review).item()
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    label = "Good Product" if avg_score > 0.5 else "Not a Good Product"

    return {"product_id": product_id, "score": round(avg_score, 2), "label": label}

# API Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    product_id = data.get("product_id")

    if not product_id:
        return jsonify({"error": "Missing product_id"}), 400

    result = predict_product(product_id)
    return jsonify(result)

# UI Route
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
