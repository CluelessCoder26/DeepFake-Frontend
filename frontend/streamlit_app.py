import streamlit as st
import requests

# =====================================
# CONFIG
# =====================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🎭",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .stMetric {
        border-radius: 10px;
        padding: 10px;
        background-color: #f5f5f5;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================
# HEADER
# =====================================

st.title("🎭 DeepFake Detection System")

st.markdown(
    """
    ### CNN + Vision Transformer Hybrid Model

    Detect whether an uploaded image or video
    is REAL or FAKE using a hybrid AI model
    with Grad-CAM explainability.
    """
)

# =====================================
# SIDEBAR
# =====================================

option = st.sidebar.selectbox(

    "Select Detection Type",

    [
        "Image Detection",
        "Video Detection"
    ]
)

# =====================================
# IMAGE DETECTION
# =====================================

if option == "Image Detection":

    st.header("🖼️ Image DeepFake Detection")

    uploaded_file = st.file_uploader(

        "Upload Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                uploaded_file,
                caption="Uploaded Image",
                use_container_width=True
            )

        if st.button(
            "Analyze Image"
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                file_bytes = uploaded_file.getvalue()

                files = {

                    "file": (
                        uploaded_file.name,
                        file_bytes,
                        uploaded_file.type
                    )
                }

                # =========================
                # IMAGE PREDICTION
                # =========================

                prediction_response = requests.post(

                    f"{API_URL}/predict-image",

                    files=files
                )

                prediction_result = (
                    prediction_response.json()
                )

                prediction = (
                    prediction_result["prediction"]
                )

                confidence = (
                    prediction_result["confidence"]
                )

                # =========================
                # GRADCAM
                # =========================

                gradcam_response = requests.post(

                    f"{API_URL}/gradcam",

                    files=files
                )

                gradcam_result = (
                    gradcam_response.json()
                )

            with col2:

                st.image(

                    f"{API_URL}"
                    f"{gradcam_result['gradcam_image']}",

                    caption="GradCAM Heatmap",

                    use_container_width=True
                )

            st.divider()

            if prediction == "FAKE":

                st.error(
                    "⚠️ FAKE IMAGE DETECTED"
                )

            else:

                st.success(
                    "✅ REAL IMAGE DETECTED"
                )

            st.progress(
                int(confidence)
            )

            st.metric(
                "Confidence",
                f"{confidence}%"
            )

# =====================================
# VIDEO DETECTION
# =====================================

elif option == "Video Detection":

    st.header("🎥 Video DeepFake Detection")

    uploaded_video = st.file_uploader(

        "Upload Video",

        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded_video is not None:

        st.video(
            uploaded_video
        )

        if st.button(
            "Analyze Video"
        ):

            with st.spinner(
                "Analyzing video..."
            ):

                file_bytes = (
                    uploaded_video.getvalue()
                )

                files = {

                    "file": (
                        uploaded_video.name,
                        file_bytes,
                        uploaded_video.type
                    )
                }

                response = requests.post(

                    f"{API_URL}/predict-video",

                    files=files
                )

                result = response.json()

            prediction = (
                result["prediction"]
            )

            confidence = (
                result["confidence"]
            )

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(

                    "Faces Processed",

                    result[
                        "faces_processed"
                    ]
                )

            with col2:

                st.metric(

                    "Fake Frames",

                    result[
                        "fake_frames"
                    ]
                )

            with col3:

                st.metric(

                    "Real Frames",

                    result[
                        "real_frames"
                    ]
                )

            st.divider()

            if prediction == "FAKE":

                st.error(
                    "⚠️ FAKE VIDEO DETECTED"
                )

            else:

                st.success(
                    "✅ REAL VIDEO DETECTED"
                )

            st.progress(
                int(confidence)
            )

            st.metric(
                "Confidence",
                f"{confidence}%"
            )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown(

    """
    DeepFake Detection System

    EfficientNet-B0 + ViT-Small Hybrid Model

    FastAPI + Streamlit + GradCAM
    """
)