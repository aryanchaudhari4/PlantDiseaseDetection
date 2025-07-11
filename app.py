import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# Load label classes
with open("model/label_classes.pkl", "rb") as f:
    class_names = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detection from Leaf Image")
st.write("Upload a clear image of a leaf to predict its disease class.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and normalize
    resized_image = cv2.resize(image_rgb, (128, 128))
    normalized_image = resized_image / 255.0
    input_data = np.expand_dims(normalized_image, axis=0)

    # Predict
    prediction = model.predict(input_data)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show results
    st.image(image_rgb, caption="Uploaded Leaf Image", use_container_width=True)
    st.markdown(f"### 🧪 Prediction: **{predicted_class}**")
    st.markdown(f"🔢 Confidence: **{confidence:.2f}%**")
