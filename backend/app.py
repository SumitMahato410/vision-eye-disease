import io
import os
import numpy as np
import joblib

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from knowledge import SYMPTOMS, KEYS, INFO
from train_model import train
from image_analysis import analyze


# --------------------------------------------------
# MODEL SETUP
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.pkl"
)

# Train the model automatically if model.pkl doesn't exist
if not os.path.exists(MODEL_PATH):
    train(MODEL_PATH)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FLASK APP
# --------------------------------------------------

app = Flask(__name__)

# Allow frontend (Vercel) to communicate with backend (Render)
CORS(app)

# Maximum uploaded image size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Symptoms detected automatically from the uploaded image
AUTO = ("clear_eye", "red_eye")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return jsonify(
        status="success",
        message="Vision Eye Disease Detection API is running"
    )


@app.get("/health")
def health():
    return jsonify(
        status="healthy"
    )


# --------------------------------------------------
# GET AVAILABLE SYMPTOMS
# --------------------------------------------------

@app.get("/api/symptoms")
def symptoms():
    return jsonify([
        {
            "key": k,
            "label": v
        }
        for k, v in SYMPTOMS.items()
        if k not in AUTO
    ])


# --------------------------------------------------
# DISEASE PREDICTION
# --------------------------------------------------

@app.post("/api/predict")
def predict():

    # Get uploaded image
    f = request.files.get("image")

    if f is None:
        return jsonify(
            error="Please upload an eye image."
        ), 400

    # Open image
    try:
        img = Image.open(
            io.BytesIO(f.read())
        )

    except Exception:
        return jsonify(
            error="That file is not a valid image."
        ), 400

    # Analyze image
    try:
        res = analyze(img)

    except Exception as e:
        return jsonify(
            error=f"Image analysis failed: {str(e)}"
        ), 500

    # Check image analysis error
    if res["error"]:
        return jsonify(
            error=res["error"]
        ), 422

    # Get manually selected symptoms
    extra = [
        s
        for s in request.form.get(
            "symptoms",
            ""
        ).split(",")
        if s in SYMPTOMS and s not in AUTO
    ]

    # Combine manually selected symptoms
    # with automatically detected image symptom
    chosen = extra + [res["symptom"]]

    # Create feature vector
    x = np.array([
        [
            1 if k in chosen else 0
            for k in KEYS
        ]
    ])

    # Predict probabilities
    proba = model.predict_proba(x)[0]

    # Get top 3 predictions
    order = np.argsort(proba)[::-1][:3]

    results = [
        {
            "disease": model.classes_[i],
            "confidence": round(
                float(proba[i]) * 100,
                1
            )
        }
        for i in order
    ]

    # Get information about top disease
    urgency, desc, prevention = INFO[
        results[0]["disease"]
    ]

    # Return prediction
    return jsonify(
        top=results[0],
        others=results[1:],
        urgency=urgency,
        description=desc,
        prevention=prevention,
        uncertain=results[0]["confidence"] < 50,
        image=res,
        disclaimer=(
            "This is an AI screening tool, "
            "not a medical diagnosis. "
            "Please consult an eye doctor."
        )
    )


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return jsonify(
        error="Image is too large. Maximum size is 10 MB."
    ), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify(
        error="API endpoint not found."
    ), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify(
        error="Internal server error."
    ), 500


# --------------------------------------------------
# LOCAL DEVELOPMENT
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )