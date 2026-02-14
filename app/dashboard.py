import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Industrial Defect Detection",
    layout="centered"
)

st.title("🔍 Industrial Surface Defect Detection")
st.markdown("Upload a metal surface image to detect defects.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            if result["status"] == "success":
                st.success(f"Predicted: {result['predicted_class']}")
                st.metric("Confidence", f"{result['confidence']:.4f}")

            else:
                st.warning("Model is uncertain.")
                st.metric("Confidence", f"{result['confidence']:.4f}")

        else:
            st.error("Error communicating with backend.")
