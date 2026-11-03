import pickle
from pathlib import Path
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "sentiment_model.pkl"

app = Flask(__name__)
model = None

def load_model():
    global model
    if MODEL_FILE.exists():
        with open(MODEL_FILE, "rb") as file:
            model = pickle.load(file)

load_model()

@app.route("/")
def home():
    return render_template("index.html", model_ready=model is not None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not found. Please run python train_model.py first."}), 503

    data = request.get_json(silent=True) or request.form
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_
    confidence = float(max(probabilities)) * 100

    return jsonify({
        "sentiment": prediction.title(),
        "confidence": round(confidence, 2),
        "scores": {str(label).title(): round(float(score) * 100, 2)
                   for label, score in zip(classes, probabilities)}
    })

if __name__ == "__main__":
    app.run(debug=True)
