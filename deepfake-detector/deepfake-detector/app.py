"""Step 4: Streamlit demo.   Run with:  streamlit run app.py"""
import json

import numpy as np
import streamlit as st
from PIL import Image

import config
import inference

st.set_page_config(page_title="DeepFake Image Detection", page_icon="🔍", layout="wide")
st.title("🔍 DeepFake Image Detection")
st.caption("EfficientNetB4 transfer learning + Grad-CAM visual explanation")


@st.cache_resource(show_spinner="Loading model...")
def load():
    return inference.load_artifacts()


try:
    model, grad_model, img_size = load()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Model performance (test set)")
    mpath = config.OUTPUT_DIR / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        st.metric("Accuracy", f"{m['accuracy']:.2%}")
        st.metric("Precision", f"{m['precision']:.2%}")
        st.metric("Recall", f"{m['recall']:.2%}")
        st.metric("F1-score", f"{m['f1']:.2%}")
    else:
        st.info("Run train.py to generate metrics.")

file = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])
if file:
    img = np.array(Image.open(file).convert("RGB"))
    with st.spinner("Analyzing..."):
        res = inference.analyze(grad_model, img, img_size)

    if res["label"] == "DEEPFAKE":
        st.error(f"### Prediction: DEEPFAKE  —  {res['confidence']:.1%} confidence")
    else:
        st.success(f"### Prediction: REAL  —  {res['confidence']:.1%} confidence")
    st.progress(float(res["p_fake"]), text=f"Probability of being fake: {res['p_fake']:.1%}")

    c1, c2 = st.columns(2)
    c1.subheader("Input image")
    c1.image(res["resized"])
    c2.subheader("Regions influencing the decision")
    c2.image(res["overlay"])
    st.caption("Red/yellow = regions the model relied on most. This is an explanation of model "
               "attention, not proof that those regions were manipulated.")
else:
    st.info("Upload an image to get a prediction.")
