import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
from PIL import Image
import os

# ---------------------------
# Load YOLO model
# ---------------------------
MODEL_PATH = "best.pt"  # Place your model file in the same directory
model = YOLO(MODEL_PATH)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="YOLOv8 Video Object Detection", layout="wide")
st.title("🎥 YOLOv8 Video Object Detection App")

st.markdown(
    """
    Upload a video file below, and the app will run object detection frame-by-frame using your YOLOv8 model.  
    💡 Works best with short videos (<30 seconds) for cloud deployment.
    """
)

uploaded_video = st.file_uploader("📁 Upload a video", type=["mp4", "mov", "avi", "mkv"])

# ---------------------------
# Process video if uploaded
# ---------------------------
if uploaded_video:
    # Save video temporarily
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)

    st.write("🔍 Running YOLOv8 detection...")
    output_path = os.path.join(temp_dir, "output.mp4")

    # OpenCV video processing
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        count += 1
        progress_bar.progress(count / frame_count)

    cap.release()
    out.release()

    st.success("✅ Detection complete!")
    st.video(output_path)
    st.download_button("⬇️ Download processed video", open(output_path, "rb"), file_name="detected_output.mp4")

else:
    st.info("Please upload a video file to start detection.")
