
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Eye Disease Classification")
st.write(
    "An educational machine-learning project using "
    "MobileNetV2 and the RFMiD retinal image dataset."
)

MODEL_PATH = Path(__file__).parent / "eye_disease_classifier.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please add "
        "eye_disease_classifier.keras to the repository."
    )
    st.stop()

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded retinal image", use_container_width=True)

    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing image..."):
            resized_image = image.resize((224, 224))
            image_array = np.array(resized_image, dtype=np.float32)

            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                image_array
            )
            image_array = np.expand_dims(image_array, axis=0)

            score = float(model.predict(image_array, verbose=0)[0][0])

        st.subheader("Model output")
        st.metric("Disease_Risk = 0 score", f"{(1-score)*100:.2f}%")
        st.metric("Disease_Risk = 1 score", f"{score*100:.2f}%")

        st.info(
            "These are model scores for the dataset labels, "
            "not a medical diagnosis or validated clinical probabilities."
        )

st.divider()
st.caption(
    "Educational demonstration only. Consult a qualified eye-care "
    "professional for medical concerns."
)
