import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

import os
import gdown

# Create model folder
os.makedirs("model", exist_ok=True)

# File paths
keras_path = "model/BestModel.keras"
h5_path = "model/BestModel.h5"
task_path = "model/face_landmarker.task"

# Download if not exists
if not os.path.exists(keras_path):
    gdown.download("https://drive.google.com/uc?id=1Ps6p2Td0IyeCJWOXU60lMuVXe_oMFZWw", keras_path, quiet=False)

if not os.path.exists(h5_path):
    gdown.download("https://drive.google.com/uc?id=1i1ixpBm0NOeCaWeHslAjF8Rksjb_YLYg", h5_path, quiet=False)

if not os.path.exists(task_path):
    gdown.download("https://drive.google.com/uc?id=1rRK038Sua8khNDlU_ZOBqehuScHjzseM", task_path, quiet=False)

# Load model
from tensorflow.keras.models import load_model
model = load_model(keras_path, compile=False)
# =========================
# Page Config
# =========================
st.set_page_config(page_title="💄 AI Lip Try-On", layout="centered")

# =========================
# Stylish UI
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b, #3b0764);
    color: white;
}
h1 {
    text-align: center;
    color: #ff4b91;
}
.color-circle {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: inline-block;
    margin: 5px;
    border: 2px solid white;
}
</style>
""", unsafe_allow_html=True)

st.title("💄 AI Lip Try-On System")
st.markdown("### Try Lipstick • Balm • Tint • Liquid Lipstick")

# =========================
# Mediapipe
# =========================
@st.cache_resource
def load_mediapipe():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    )

face_mesh = load_mediapipe()

# =========================
# 💋 Lip Product Types
# =========================
lip_types = [
    "💄 Lipstick",
    "💧 Liquid Lipstick",
    "🌸 Lip Balm",
    "🔥 Lip Tint",
    "✨ Lip Plumper",
    "✏️ Lip Liner"
]

lip_type = st.selectbox("Select Lip Product", lip_types)

# =========================
# 💄 Lipstick Palettes (GROUPED)
# =========================
lipstick_palettes = {

    "💋 Classic Reds": {
        "Ruby Red": (30, 30, 200),
        "Cherry": (40, 40, 180),
        "Crimson": (20, 20, 150),
    },

    "🌸 Pink Collection": {
        "Baby Pink": (200, 160, 255),
        "Rose": (180, 120, 220),
        "Hot Pink": (180, 0, 255),
    },

    "🌿 Nude": {
        "Peach Nude": (140, 170, 200),
        "Nude Beige": (160, 180, 200),
        "Soft Brown": (80, 100, 140),
    },

    "🔥 Bold": {
        "Fuchsia": (180, 0, 255),
        "Magenta": (150, 0, 200),
        "Neon Red": (10, 10, 255),
    }
}

# =========================
# Palette Selection
# =========================
palette = st.selectbox("Choose Palette", list(lipstick_palettes.keys()))
shade = st.selectbox("Choose Shade", list(lipstick_palettes[palette].keys()))
selected_color = lipstick_palettes[palette][shade]

# =========================
# 🎨 Shade Preview
# =========================
circles_html = ""
for name, color in lipstick_palettes[palette].items():
    circles_html += f'<div class="color-circle" style="background-color: rgb({color[2]},{color[1]},{color[0]});"></div>'

st.markdown(circles_html, unsafe_allow_html=True)

opacity = st.slider("Opacity", 0.2, 0.9, 0.5)

# =========================
# Input
# =========================
source = st.radio("Input Source", ["Camera", "Upload"], horizontal=True)

img_file = (
    st.camera_input("Take Photo")
    if source == "Camera"
    else st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
)

# =========================
# Lip Mask
# =========================
LIPS = [61,185,40,39,37,0,267,269,270,409,
        291,375,321,405,314,17,84,181,91,146]

def get_lip_mask(image):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    pts = np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in LIPS])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    return mask

# =========================
# Apply Lip Product
# =========================
def apply_lip_product(image, mask, color, alpha, product_type):

    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = mask.astype(np.float32)/255.0
    mask_3d = np.stack([mask]*3, axis=2)

    img = image.astype(np.float32)
    color_layer = np.full_like(img, color, dtype=np.float32)

    if product_type == "💧 Liquid Lipstick":
        alpha += 0.2
    elif product_type == "🌸 Lip Balm":
        alpha = 0.2
    elif product_type == "🔥 Lip Tint":
        alpha = 0.3
    elif product_type == "✨ Lip Plumper":
        alpha = 0.4
    elif product_type == "✏️ Lip Liner":
        edges = cv2.Canny(mask.astype(np.uint8)*255, 50, 150)
        edge_mask = edges.astype(np.float32)/255.0
        edge_mask = np.stack([edge_mask]*3, axis=2)
        return np.clip(img*(1-edge_mask) + color_layer*edge_mask, 0, 255).astype(np.uint8)

    blended = img*(1 - mask_3d*alpha) + color_layer*(mask_3d*alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)

# =========================
# MAIN
# =========================
if img_file:

    image = Image.open(img_file).convert("RGB")
    image = np.array(image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Original", use_container_width=True)

    mask = get_lip_mask(image_bgr)

    if mask is None:
        st.error("❌ Lips not detected")
        st.stop()

    result = apply_lip_product(image_bgr, mask, selected_color, opacity, lip_type)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    st.image(result_rgb, caption=f"{lip_type} - {shade}", use_container_width=True)

    st.success("✨ Lip product applied successfully!")