from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract
import io
from gtts import gTTS
import os
import uuid

app = Flask(__name__)
CORS(app)

# Load object detection model
MODEL_PATH = "frozen_inference_graph.pb"
CONFIG_PATH = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
LABELS_PATH = "lables"

with open(LABELS_PATH, "r") as f:
    class_names = f.read().rstrip("\n").split("\n")

detector = cv2.dnn_DetectionModel(MODEL_PATH, CONFIG_PATH)
detector.setInputSize(320, 320)
detector.setInputScale(1.0 / 127.5)
detector.setInputMean((127.5, 127.5, 127.5))
detector.setInputSwapRB(True)

# Load image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/detect', methods=['POST'])
def detect_objects():
    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({"error": "No image uploaded"}), 400

    np_img = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    class_ids, confs, boxes = detector.detect(img, confThreshold=0.5)
    results = []

    if len(class_ids) > 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), boxes):
            label = class_names[class_id - 1]
            results.append({"label": label, "confidence": float(confidence), "box": box.tolist()})

    return jsonify({"objects": results})
    
@app.route('/')
def home():
    return "Flask API is running! Use /detect, /caption, /ocr, or /speak."

@app.route('/caption', methods=['POST'])
def caption_image():
    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(image_file).convert('RGB')
    inputs = processor(images=img, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return jsonify({"caption": caption})


@app.route('/ocr', methods=['POST'])
def perform_ocr():
    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)

    return jsonify({"text": text.strip()})


@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    filename = f"speech_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("static", filename)
    tts = gTTS(text)
    tts.save(filepath)

    return jsonify({"audio_url": f"/static/{filename}"})


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
