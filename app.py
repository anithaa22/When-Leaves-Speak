import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# -----------------------------
# Load your trained sugarcane model
# -----------------------------
model = load_model("sugarcane_mobilenet.h5")  # make sure this file is in the same folder

# -----------------------------
# Define sugarcane disease classes
# -----------------------------
class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# Define treatment/medicine recommendations for each disease
treatment_dict = {
    'Healthy': "No treatment needed.",
    'Mosaic': "Use virus-free planting material. Remove infected plants.",
    'RedRot': "Use Carbendazim 50% WP - 1 gram per liter (500g per acre).",
    'Rust': "Spray Propiconazole 25% EC - 1 ml per liter (400 ml per acre).",
    'Yellow': "Spray Mancozeb 75% WP - 2 grams per liter."
}

# -----------------------------
# Confidence threshold
# -----------------------------
confidence_threshold = 0.4  # Adjust to capture all sugarcane leaves

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Sugarcane Leaf Disease Detection 🌱")
st.write("Upload a sugarcane leaf image. The model will predict the disease and suggest treatment.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    pred_class_index = np.argmax(predictions)
    pred_confidence = predictions[0][pred_class_index]

    # Show results based on confidence
    if pred_confidence < confidence_threshold:
        st.warning("Disease not recognized clearly. Please consult an agricultural expert.")
    else:
        pred_class_name = class_names[pred_class_index]
        st.success(f"Predicted Disease: {pred_class_name}")
        st.info(f"Confidence: {pred_confidence*100:.2f}%")
        st.write(f"Recommended Treatment: {treatment_dict[pred_class_name]}")

    # Optional: Show probability for all classes
    st.subheader("All Class Probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name} : {predictions[0][i]*100:.2f} %")

