"""
Flask API server for Metacognitive Attention Model
Handles image upload, processing, and saliency visualization
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
from pathlib import Path
import logging

from attention_model import FullIttiKochModel, visualize_attention

# Configuration
UPLOAD_FOLDER = Path("uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Setup
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

UPLOAD_FOLDER.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Load model
try:
    model = FullIttiKochModel().to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image(image_array: np.ndarray) -> str:
    """Encode numpy array as base64 string"""
    _, buffer = cv2.imencode(".png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


def process_image(file_path: str) -> dict:
    """
    Process image and generate saliency visualization

    Returns:
        dict with original, saliency, heatmap, and overlay images
    """
    try:
        # Load image
        img = Image.open(file_path).convert("RGB")

        # Resize for processing if too large (maintain aspect ratio)
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Get final display size and array
        original_size = img.size
        img_array = np.array(img)

        # Preprocess for model
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).to(DEVICE)

        # Generate saliency
        with torch.no_grad():
            saliency = model(img_tensor)

        # Process saliency map
        s_map = saliency[0, 0].cpu().numpy()
        s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min() + 1e-8)

        # Resize to original dimensions
        s_map_resized = cv2.resize(s_map, original_size)
        gray_saliency = (s_map_resized * 255).astype(np.uint8)

        # Apply colormap
        heatmap = cv2.applyColorMap(gray_saliency, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = cv2.addWeighted(
            cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0
        )
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        return {
            "success": True,
            "original": encode_image(img_array),
            "saliency": encode_image(gray_saliency),
            "heatmap": encode_image(heatmap),
            "overlay": encode_image(overlay),
            "shape": list(img_array.shape),
            "device": DEVICE,
        }

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"success": False, "error": str(e)}


@app.route("/", methods=["GET"])
def index():
    """Serve the test interface"""
    return send_file("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "device": DEVICE, "model_loaded": True})


@app.route("/upload", methods=["POST"])
def upload_image():
    """Upload and process image"""
    try:
        # Check if file present
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format"}), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = app.config["UPLOAD_FOLDER"] / filename
        file.save(filepath)

        # Process image
        result = process_image(str(filepath))

        # Cleanup
        filepath.unlink()

        return jsonify(result)

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze image from base64 or file"""
    try:
        data = request.get_json()

        if not data:
            # Try file upload
            return upload_image()

        # Handle base64 image
        if "image" in data:
            import base64

            image_data = data["image"]
            if "," in image_data:
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Save temporarily
            temp_path = app.config["UPLOAD_FOLDER"] / "temp.png"
            img.save(temp_path)

            result = process_image(str(temp_path))
            temp_path.unlink()

            return jsonify(result)

        return jsonify({"error": "No image data provided"}), 400

    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large"""
    return jsonify({"error": "File too large"}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
