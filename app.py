import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("sugarcane_mobilenet.h5")  # change name if different

class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

medicine_dict = {
    "Healthy": "No treatment needed. Maintain proper irrigation and nutrients.",
    "Mosaic": "Use virus-free planting material. Spray Imidacloprid 0.3 ml per liter.",
    "RedRot": "Use Carbendazim 50% WP - 1 gram per liter (500g per acre).",
    "Rust": "Spray Mancozeb 2.5 grams per liter.",
    "Yellow": "Apply Nitrogen fertilizer - 50 kg per acre."
}

st.title("🌱 When Leaves Speak")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

   predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
confidence = float(np.max(predictions)) * 100

class_names = [
    'Healthy',
    'Mosaic',
    'Red Rot',
    'Rust',
    'Yellow Leaf'
]

# Recommended treatments for sugarcane diseases
treatments = {
    'Healthy': "No treatment needed.",
    'Mosaic': "Remove affected leaves, use resistant varieties.",
    'Red Rot': "Use Carbendazim 50% WP - 1 gram per liter (500g per acre).",
    'Rust': "Apply fungicides like Propiconazole 25% EC.",
    'Yellow Leaf': "Spray Nitrogen fertilizer + remove infected canes."
}

# Threshold to decide if image is actually sugarcane
if confidence < 80:
    st.error("⚠ This is NOT a sugarcane leaf.")
else:
    disease = class_names[predicted_class]
    st.success(f"Predicted Disease: {disease}")
    st.success(f"Confidence: {confidence:.2f}%")
    st.info(f"Recommended Treatment: {treatments[disease]}")

    
