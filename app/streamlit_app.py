"""
streamlit_app.py

PKR Fake Currency Detector — Live Dashboard (Simple UI)
Upload a currency note image OR take a live photo, manually crop to
frame just the note, then get a Fake / Real / Not A Note prediction
with confidence score, a low-confidence safety warning, OCR-based
keyword detection for known fake/novelty note text, and a Grad-CAM
heatmap explaining the decision.

Run with:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from tensorflow.keras.models import load_model
import pytesseract

sys.path.append(str(Path(__file__).parent.parent / "src"))
from gradcam import make_gradcam_heatmap, overlay_heatmap  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Resolved relative to this file's location (not the working directory),
# so this works both locally (run from app/) AND on Streamlit Cloud
# (which runs from the repo root) without needing to change anything.
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "mobilenetv2_3class_currency.keras")
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "out_relu"
BASE_MODEL_NAME = "mobilenetv2_1.00_224"
CONFIDENCE_THRESHOLD = 0.85

# Must match class_indices order from training: {'Fake Notes': 0, 'Not A Note': 1, 'Real Notes': 2}
CLASS_NAMES = ["Fake Notes", "Not A Note", "Real Notes"]

import shutil

# Point pytesseract to the Tesseract installation.
# On Streamlit Cloud (Linux), Tesseract is installed via packages.txt and
# is already on the system PATH, so shutil.which finds it automatically.
# On Windows locally, it falls back to the default install location.
_tesseract_path = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = _tesseract_path

# Known phrases that indicate a novelty/play-money/specimen note, not genuine currency.
# If OCR finds any of these, we override the CNN's decision to "Fake Notes"
# regardless of how confident the CNN was, since explicit text evidence is
# stronger than visual pattern-matching for this specific failure case.
#
# NOTE: Some phrases (e.g. "bachon ka khel") are printed in Urdu SCRIPT on
# the actual notes, not Roman letters - so we need the matching Urdu-script
# text here, and Tesseract needs the Urdu language pack (lang="eng+urd")
# to be able to read it at all.
FAKE_INDICATOR_PHRASES = [
    "bachon ka khel",       # Roman-script variant, in case OCR transliterates
    "بچوں کا کھیل",         # Urdu script: "bachon ka khel"
    "پیارے بچے",
    "من کے سچے",            # Urdu script: "pyaare bache" (from "pyaare bache sab se achay")
    "specimen",
    "sample",
    "replica",
    "not legal tender",
    "novelty",
    "play money",
    "toy money",
]

# Repeated-zero serial numbers (e.g. "0000000000") are a common marker on
# specimen/novelty notes. If the OCR text contains a run of 6+ zeros in a
# row, treat it as a fake indicator too.
import re

def has_all_zero_serial(text: str) -> bool:
    return bool(re.search(r"0{6,}", text))

st.set_page_config(page_title="PKR Currency Detector", page_icon="💵", layout="centered")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_trained_model(model_path: str):
    return load_model(model_path)


def preprocess_image(image: Image.Image, target_size=IMG_SIZE) -> np.ndarray:
    image = image.convert("RGB").resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


def check_fake_indicator_text(image: Image.Image):
    """
    Runs OCR (English + Urdu) on the image and checks the extracted text
    against a list of known fake/novelty note phrases, plus a check for
    all-zero serial numbers (a common specimen/novelty marker).

    Returns:
        (extracted_text, matched_phrases)
    """
    try:
        # lang="eng+urd" reads BOTH English and Urdu script in one pass.
        # Requires urd.traineddata to be installed in Tesseract's tessdata folder.
        extracted_text = pytesseract.image_to_string(image.convert("RGB"), lang="eng+urd").strip()
    except Exception:
        return "", []

    matched = [phrase for phrase in FAKE_INDICATOR_PHRASES if phrase in extracted_text.lower() or phrase in extracted_text]

    if has_all_zero_serial(extracted_text):
        matched.append("all-zero serial number")

    return extracted_text, matched


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("💵 PKR Currency Detector")
st.write(
    "AI-powered authenticity check for Pakistani Rupee notes, powered by "
    "MobileNetV2 with Grad-CAM explainability and OCR-based text verification."
)

st.info(
    "🔬 Research prototype — demonstrates AI-based authenticity screening across "
    "3 classes (Real, Fake, Not A Note). Always verify currency through official "
    "channels for financial decisions. Visual similarity alone cannot detect all "
    "security features (UV ink, watermarks, embedded threads)."
)

# --- Input method selector: Upload or Live Camera ---
input_method = st.radio("Choose input method:", ["📁 Upload Image", "📷 Use Camera"], horizontal=True)

raw_image = None

if input_method == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a currency note image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Take a photo of the currency note")
    if camera_file is not None:
        raw_image = Image.open(camera_file)

image = None

if raw_image is not None:
    st.subheader("✂️ Frame the Note")
    st.caption(
        "Drag the corners of the box below so it tightly frames just the currency "
        "note — this removes background clutter before analysis."
    )
    image = st_cropper(raw_image, realtime_update=True, box_color="#00c9a7", aspect_ratio=None)

    st.subheader("✅ Cropped Preview")
    st.image(image, use_container_width=True)


if image is not None:
    try:
        model = load_trained_model(MODEL_PATH)
    except (OSError, IOError):
        st.error(
            "⚠️ Model file not found. Train the 3-class model first, then update "
            "MODEL_PATH at the top of this file to match your saved filename."
        )
        st.stop()

    if st.button("🔍 Analyze Note", use_container_width=True):

        with st.spinner("Analyzing..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)[0]  # array of 3 probabilities
            predicted_idx = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(predictions[predicted_idx])

            # --- OCR-based keyword override check ---
            extracted_text, matched_phrases = check_fake_indicator_text(image)

        st.subheader("🧾 Result")

        # --- OCR override takes priority over the CNN's own decision ---
        if matched_phrases:
            st.error(
                f"🚨 Suspicious text detected: **{', '.join(matched_phrases)}**. "
                f"This overrides the visual classification — likely a novelty/fake note."
            )
            predicted_class = "Fake Notes"
            st.markdown("### ❌ FAKE NOTE (flagged by text detection)")
        else:
            # --- Low-confidence safety warning ---
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ Low confidence result ({confidence*100:.1f}%). This note may closely "
                    f"resemble another category. Please verify through official channels "
                    f"(bank, ATM scanner, or manual security feature check) before relying "
                    f"on this result."
                )

            if predicted_class == "Real Notes":
                st.markdown("### ✅ REAL NOTE")
            elif predicted_class == "Fake Notes":
                st.markdown("### ❌ FAKE NOTE")
            else:
                st.markdown("### ⚠️ NOT A CURRENCY NOTE")

        st.write("**Confidence Breakdown**")
        for name, prob in zip(CLASS_NAMES, predictions):
            st.write(f"{name}: {prob*100:.1f}%")
            st.progress(float(prob))

        # --- OCR extracted text display (always shown, informational) ---
        st.subheader("📄 OCR Text Detection")
        st.caption("Text extracted from the image — used for the fake-phrase check above.")
        if extracted_text:
            st.code(extracted_text, language=None)
        else:
            st.caption("No readable text detected in this image.")

        # --- Grad-CAM (skipped for "Not A Note" predictions) ---
        if predicted_class != "Not A Note":
            st.subheader("🔬 Model Focus (Grad-CAM)")
            st.caption(
                "Highlighted regions show what the model focused on to reach this "
                "decision. Red/yellow = high attention, blue = low attention."
            )
            try:
                original_np = np.array(image.convert("RGB").resize(IMG_SIZE))
                heatmap = make_gradcam_heatmap(
                    img_array, model,
                    base_model_name=BASE_MODEL_NAME,
                    last_conv_layer_name=LAST_CONV_LAYER,
                    class_index=predicted_idx,
                )
                overlayed = overlay_heatmap(original_np, heatmap)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_np, caption="Cropped Input", use_container_width=True)
                with col2:
                    st.image(overlayed, caption="Grad-CAM Overlay", use_container_width=True)
            except Exception as e:
                st.info(f"Grad-CAM visualization unavailable: {e}")
        else:
            st.info("ℹ️ Grad-CAM is skipped for 'Not A Note' predictions since there's no currency feature to explain.")

st.markdown("---")
st.caption("AI/ML Internship Capstone Project | Malik Faisal Mukhtar | ZYNVEX-CERT-1130")