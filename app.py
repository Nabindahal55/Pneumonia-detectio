# ----------------------------
# Imports and Initial Setup
# ----------------------------
from flask import Flask, render_template, request  # Flask core components
import os
from werkzeug.utils import secure_filename  # For secure file saving
from PIL import Image  # For image processing
import numpy as np
import torch  # PyTorch
from torchvision import transforms  # PyTorch image transforms
from transformers import ViTForImageClassification, ViTConfig  # ViT model
import tensorflow as tf  # TensorFlow for Keras model

# ----------------------------
# Flask Configuration
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Upload path for user images
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size = 16MB

# ----------------------------
# Load Pretrained Models
# ----------------------------

# Load Keras DenseNet model (trained and saved in .h5 format)
keras_model = tf.keras.models.load_model("models/model.h5")

# Load Vision Transformer (ViT) from Hugging Face with modified label size
vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
vit_config.num_labels = 3
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=vit_config)
vit_model.load_state_dict(torch.load("models/vit_chest_xray_model.pth", map_location=torch.device('cpu')))
vit_model.eval()  # Set to evaluation mode

# Class labels for prediction
class_names = ['NORMAL', 'NOT_CHEST', 'PNEUMONIA']

# ----------------------------
# Preprocessing Functions
# ----------------------------

# PyTorch transform pipeline for ViT
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Preprocessing for DenseNet (Keras)
def preprocess_for_keras(image_path):
    img = Image.open(image_path).convert('RGB').resize((250, 250))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Preprocessing for ViT (PyTorch)
def preprocess_for_vit(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = vit_transform(img).unsqueeze(0)  # Add batch dimension
    return tensor

# ----------------------------
# Route: Home Page
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize variables
    image_path = None
    error = None
    keras_pred = None
    vit_pred = None
    prediction = None
    model_used = None
    uploaded = False

    # Handle file upload
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == '':
            error = "No file selected."
        elif not allowed_file(file.filename):
            error = "Invalid file format. Use JPG or PNG."
        else:
            try:
                filename = secure_filename(file.filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                uploaded = True
            except Exception as e:
                error = f"Upload failed: {e}"

    # Render home page
    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        image_path=image_path,
        keras_pred=keras_pred,
        vit_pred=vit_pred,
        model_used=model_used,
        uploaded=uploaded,
        keras_conf=None,
        vit_conf=None
    )

# ----------------------------
# Route: Predict from Image
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Extract form inputs
    image_path = request.form.get("image_path")
    model_choice = request.form.get("model")
    keras_pred = None
    vit_pred = None
    prediction = None
    model_used = None
    error = None
    keras_conf = None
    vit_conf = None

    try:
        # ---- DenseNet prediction ----
        if model_choice == "densenet":
            img = preprocess_for_keras(image_path)
            preds = keras_model.predict(img, verbose=0)[0]
            pred_idx = np.argmax(preds)
            keras_conf = preds[pred_idx] * 100
            prediction = f"{class_names[pred_idx]} ({keras_conf:.2f}%)"
            model_used = "DenseNet"

        # ---- ViT prediction ----
        elif model_choice == "vit":
            img_tensor = preprocess_for_vit(image_path)
            with torch.no_grad():
                outputs = vit_model(img_tensor)
                probs = torch.softmax(outputs.logits, dim=1)
                conf, pred_class = torch.max(probs, dim=1)
                vit_conf = conf.item() * 100
                prediction = f"{class_names[pred_class.item()]} ({vit_conf:.2f}%)"
                model_used = "ViT"

        # ---- Combined analysis ----
        elif model_choice == "Combined analysis":
            # Predict from DenseNet
            img1 = preprocess_for_keras(image_path)
            keras_preds = keras_model.predict(img1, verbose=0)[0]
            keras_idx = np.argmax(keras_preds)
            keras_label = class_names[keras_idx]
            keras_conf = keras_preds[keras_idx] * 100
            keras_pred = f"{keras_label} ({keras_conf:.2f}%)"

            # Predict from ViT
            img2 = preprocess_for_vit(image_path)
            with torch.no_grad():
                outputs = vit_model(img2)
                probs = torch.softmax(outputs.logits, dim=1)
                vit_conf_tensor, vit_class_tensor = torch.max(probs, dim=1)
                vit_idx = vit_class_tensor.item()
                vit_label = class_names[vit_idx]
                vit_conf = vit_conf_tensor.item() * 100
                vit_pred = f"{vit_label} ({vit_conf:.2f}%)"

            # Combine result
            if keras_label == vit_label:
                avg_conf = (keras_conf + vit_conf) / 2
                color = 'green' if keras_label == 'NORMAL' else 'orange' if keras_label == 'NOT_CHEST' else 'red'
                prediction = f"<span style='color:{color}'>{keras_label} (Average Confidence: {avg_conf:.2f}%)</span>"
            else:
                prediction = f"<span style='color:red'>Model Disagreement:</span><br><strong>DenseNet:</strong> {keras_pred}<br><strong>ViT:</strong> {vit_pred}"

            model_used = "Combined"

        else:
            error = "Please select a valid model."

    except Exception as e:
        error = f"Prediction failed: {e}"

    # Render result on home page
    return render_template(
        "index.html",
        image_path=image_path,
        keras_pred=keras_pred,
        vit_pred=vit_pred,
        prediction=prediction,
        model_used=model_used,
        error=error,
        uploaded=True,
        keras_conf=keras_conf,
        vit_conf=vit_conf
    )

# ----------------------------
# Run the Flask App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)  # Run app in debug mode for development
