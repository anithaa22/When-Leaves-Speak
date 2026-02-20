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

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    st.subheader("Prediction Result")

    if confidence < 0.5:
        st.warning("Disease not recognized clearly. Please consult agricultural expert.")
    else:
        st.success(f"Disease: {predicted_class}")
        st.write(f"Confidence: {round(confidence*100,2)}%")
        st.write(f"Recommended Treatment: {medicine_dict[predicted_class]}")