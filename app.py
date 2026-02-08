from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
from collections import deque

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.secret_key = "some_random_secret_key"

DASHBOARD_PASSWORD = "81412"

history = deque(maxlen=15)

# Load models once using ONNX Runtime
def load_model(model_path):
    return ort.InferenceSession(model_path)

nets = {
    "2": ("Brain Tumor Detection", load_model("models/tumor_detection/resnet18.onnx")),
    "3": ("Brain Tumor Classification", load_model("models/tumor_classification/resnet18.onnx")),
    "4": ("Alzheimer's Classification", load_model("models/alzheimers/resnet18.onnx")),
    "5": ("COVID-19 Detection", load_model("models/covid/resnet18.onnx")),
    "6": ("Fracture Detection", load_model("models/fractures/resnet18.onnx")),
    "7": ("Eye Disease Classification", load_model("models/EDC/resnet18.onnx")),
    "8": ("Pneumonia Detection", load_model("models/pneumonia/resnet18.onnx")),
}

# Load label files
labels = {}
for key, (name, model) in nets.items():
    label_file = f"models/{name.lower().replace(' ', '_')}/labels.txt"
    if os.path.exists(label_file):
        with open(label_file) as f:
            labels[key] = [line.strip() for line in f.readlines()]
    else:
        labels[key] = []

model_enabled = {key: True for key in nets.keys()}

# Preprocess image for ONNX model
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))  # C,H,W
    img_data = np.expand_dims(img_data, 0)  # batch dimension
    return img_data

@app.route("/")
def index():
    enabled_models = {k: v for k, v in nets.items() if model_enabled.get(k, False)}
    return render_template("index.html", models=enabled_models)

@app.route("/info")
def info():
    return render_template("info.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == DASHBOARD_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error=True)
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in", False):
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/classify", methods=["POST"])
def classify():
    try:
        file = request.files["image"]
        mode = request.form["mode"]
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        if not os.path.exists(app.config["UPLOAD_FOLDER"]):
            os.makedirs(app.config["UPLOAD_FOLDER"])
        file.save(filename)

        if mode not in nets:
            return jsonify({"error": "Invalid mode"}), 400

        if not model_enabled.get(mode, False):
            return jsonify({"error": "Model is disabled"}), 400

        model_name, model = nets[mode]
        img_input = preprocess_image(filename)
        output_name = model.get_outputs()[0].name
        outputs = model.run([output_name], {"input_0": img_input})
        predicted_class = int(np.argmax(outputs[0]))
        class_desc = labels[mode][predicted_class] if labels.get(mode) else str(predicted_class)
        confidence = float(np.max(outputs[0]))

        result = {
            "model": model_name,
            "prediction": class_desc,
            "confidence": f"{confidence*100:.2f}%",
            "id": predicted_class,
            "filename": file.filename,
        }

        history.appendleft(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history")
def get_history():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(list(history))

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    history.clear()
    return jsonify({"status": "cleared"})

@app.route("/model_status", methods=["GET"])
def get_model_status():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(model_enabled)

@app.route("/toggle_model/<model_id>", methods=["POST"])
def toggle_model(model_id):
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    if model_id not in model_enabled:
        return jsonify({"error": "Invalid model id"}), 400
    model_enabled[model_id] = not model_enabled[model_id]
    return jsonify({"model_id": model_id, "enabled": model_enabled[model_id]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
