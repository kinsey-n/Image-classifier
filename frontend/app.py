import streamlit as st
import requests
import os
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.title("Image Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
    if st.button("Classify"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{BACKEND_URL}/predict", files=files)
        if response.ok:
            result = response.json()
            st.success(f"{result['label']} ({result['confidence']:.1%} confidence)")
        else:
            st.error(f"Prediction failed: {response.text}")
