import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model("sugarcane_balanced_model.h5")

# Class names
class_names = ['Healthy', 'Mosaic', 'Red Rot', 'Rust', 'Yellow Leaf']

# Disease treatments
treatments = {
    'Healthy': "No treatment needed.",
    'Mosaic': "Remove affected leaves, use resistant varieties.",
    'Red Rot': "Use Carbendazim 50% WP - 1 gram per liter (500g per acre).",
    'Rust': "Apply fungicides like Propiconazole 25% EC.",
    'Yellow Leaf': "Spray Nitrogen fertilizer + remove infected canes."
}

st.title("Sugarcane Leaf Disease Detection")

# Allow jpg, jpeg, png, webp
uploaded_file = st.file_uploader(
    "Upload a sugarcane leaf image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Preprocess the image ---
    img = img.resize((224, 224))               # resize to model input size
    img_array = np.array(img)                  # convert to array
    img_array = img_array / 255.0              # normalize
    img_array = np.expand_dims(img_array, 0)   # add batch dimension

    # --- Make prediction ---
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    # Decide sugarcane or not
    if confidence < 80:  # threshold for sugarcane detection
        st.error("⚠ This does NOT look like a sugarcane leaf.")
    else:
        disease = class_names[predicted_class]
        st.success(f"Predicted Disease: {disease}")
        st.success(f"Confidence: {confidence:.2f}%")
        st.info(f"Recommended Treatment: {treatments[disease]}")
