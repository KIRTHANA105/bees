# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import traceback
import logging
import cv2

# ------------------- Model parameters -------------------
IMG_SIZE = 224
PATCH_SIZE = 56
SEQUENCE_LENGTH = 16

# ------------------- FastAPI setup -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(filename="server.log", level=logging.INFO)

# ------------------- Load model -------------------
MODEL_PATH = os.path.join("models", "alzheimer_cnn_gru_model.h5")
model = load_model(MODEL_PATH)
logging.info(f"Loaded model from {MODEL_PATH}")
try:
    logging.info(f"Model input shape: {model.input_shape}")
except Exception:
    logging.info("Could not read model.input_shape")

# ------------------- Upload folder -------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------- Helper functions -------------------
def extract_image_patches(image, patch_size=PATCH_SIZE):
    """Extract 16 patches (4x4 grid) from image"""
    h, w = image.shape[:2]
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if i + patch_size <= h and j + patch_size <= w:
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
    # Pad or trim to SEQUENCE_LENGTH
    if len(patches) < SEQUENCE_LENGTH:
        padding = np.zeros((SEQUENCE_LENGTH - len(patches), patch_size, patch_size, 3))
        patches = np.concatenate([patches, padding], axis=0)
    elif len(patches) > SEQUENCE_LENGTH:
        patches = patches[:SEQUENCE_LENGTH]
    return np.array(patches)

# ------------------- Prediction route -------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Open image and convert to RGB
        image = Image.open(file_path).convert("RGB")
        img_np = np.array(image)

        # Resize to training image size
        img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))

        # Extract patches
        patches = extract_image_patches(img_resized, PATCH_SIZE)

        # Normalize and add batch dimension
        patch_input = np.expand_dims(patches / 255.0, axis=0)  # shape: (1,16,56,56,3)

        # Predict
        pred_prob = float(model.predict(patch_input)[0][0])
        label = "AbNormal" if pred_prob > 0.01 else "Normal"

        return JSONResponse({"prediction": label, "confidence": round(pred_prob, 4)})
    
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)

# ------------------- Main -------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
