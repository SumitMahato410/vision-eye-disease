import io, os, numpy as np, joblib
from flask import Flask, jsonify, request
from PIL import Image
from knowledge import SYMPTOMS, KEYS, INFO
from train_model import train
from image_analysis import analyze

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
if not os.path.exists(MODEL_PATH):
    train(MODEL_PATH)
model = joblib.load(MODEL_PATH)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
AUTO = ("clear_eye", "red_eye")   # detected from the image automatically

@app.get("/api/symptoms")
def symptoms():
    return jsonify([{"key": k, "label": v} for k, v in SYMPTOMS.items() if k not in AUTO])

@app.post("/api/predict")
def predict():
    f = request.files.get("image")
    if f is None: return jsonify(error="Please upload an eye image."), 400
    try: img = Image.open(io.BytesIO(f.read()))
    except Exception: return jsonify(error="That file is not a valid image."), 400
    res = analyze(img)
    if res["error"]: return jsonify(error=res["error"]), 422
    extra = [s for s in request.form.get("symptoms", "").split(",") if s in SYMPTOMS and s not in AUTO]
    chosen = extra + [res["symptom"]]
    x = np.array([[1 if k in chosen else 0 for k in KEYS]])
    proba = model.predict_proba(x)[0]
    order = np.argsort(proba)[::-1][:3]
    results = [{"disease": model.classes_[i], "confidence": round(float(proba[i]) * 100, 1)} for i in order]
    urgency, desc, prevention = INFO[results[0]["disease"]]
    return jsonify(top=results[0], others=results[1:], urgency=urgency, description=desc, prevention=prevention,
                   uncertain=results[0]["confidence"] < 50, image=res,
                   disclaimer="This is an AI screening tool, not a medical diagnosis. Please consult an eye doctor.")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
