"""Single-image prediction + Grad-CAM (used by the Streamlit app and evaluate.py)."""
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
import gradcam


def load_artifacts():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run:  python train.py")
    meta = json.loads(config.META_PATH.read_text())
    model = keras.models.load_model(config.MODEL_PATH, compile=False)
    return model, gradcam.make_gradcam_model(model), meta["img_size"]


def analyze(grad_model, img_rgb: np.ndarray, img_size: int):
    """img_rgb: HxWx3 uint8. Returns dict with label, confidence, heatmap overlay."""
    x = tf.image.resize(img_rgb, (img_size, img_size)).numpy()   # same resize as training
    cam, p_fake = gradcam.compute(grad_model, x[None])
    label = 1 if p_fake >= 0.5 else 0
    return {
        "label": config.CLASS_NAMES[label],
        "p_fake": p_fake,
        "confidence": p_fake if label else 1 - p_fake,
        "overlay": gradcam.overlay(np.uint8(x), cam),
        "resized": np.uint8(x),
    }
