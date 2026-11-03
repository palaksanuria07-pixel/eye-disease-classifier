
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

st.set_page_config(
    page_title="Eye Disease Risk Classification",
    page_icon="👁️",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "eye_disease_classifier.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("👁️ Eye Disease Risk Classification")
st.write(
    "Upload a retinal fundus image to view the model's "
    "classification result."
)

@st.cache_resource
def get_classifier():
    return load_model()

try:
    model = get_classifier()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded retinal image",
             use_container_width=True)

    if st.button("Predict", type="primary"):
        resized = image.resize((224, 224))
        array = np.asarray(resized, dtype=np.float32)
        array = preprocess_input(array)
        array = np.expand_dims(array, axis=0)

        score = float(model.predict(array, verbose=0)[0][0])
        class_0_score = 1.0 - score

        st.subheader("Prediction result")
        st.metric("Class 1 model score", f"{score * 100:.2f}%")
        st.metric("Class 0 model score", f"{class_0_score * 100:.2f}%")

        if score >= 0.5:
            st.success("Predicted category: Disease_Risk = 1")
        else:
            st.success("Predicted category: Disease_Risk = 0")

        st.caption(
            "These are model scores for the dataset labels, "
            "not clinically validated disease probabilities."
        )

st.divider()
st.subheader("Model evaluation results")
st.write("Previously reported project metrics:")
st.write("- Accuracy: 82.66%")
st.write("- Precision: 96.47%")
st.write("- Recall: 81.03%")
st.write("- AUC: 93.65%")

st.warning(
    "This is an educational research prototype, not a medical "
    "diagnostic tool. Consult an eye-care professional for "
    "medical assessment."
)
