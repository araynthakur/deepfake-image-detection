"""Grad-CAM: heatmap of the image regions that drove the model's decision."""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


def make_gradcam_model(model):
    """Model returning (last conv feature map, pre-sigmoid logit)."""
    return keras.Model(model.inputs, [model.get_layer("top_activation").output,
                                      model.get_layer("logit").output])


def compute(grad_model, x):
    """x: float array (1,H,W,3) in 0-255. Returns (heatmap 0..1, P(fake))."""
    x = tf.convert_to_tensor(x, tf.float32)
    with tf.GradientTape() as tape:
        conv, logit = grad_model(x, training=False)
        logit = logit[0, 0]
        score = logit if float(logit) >= 0 else -logit   # evidence for the predicted class
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, float(tf.sigmoid(logit))


def overlay(img_uint8, heatmap, alpha=0.45):
    """Blend a JET-colored heatmap on top of an RGB uint8 image."""
    h, w = img_uint8.shape[:2]
    cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * img_uint8 + alpha * color)
