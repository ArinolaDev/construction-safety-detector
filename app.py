import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# PIECE 6: Streamlit Deployment - Image Upload + Webcam

# Page config
st.set_page_config(
    page_title="Construction Safety Detector",
    page_icon="🦺",
    layout="wide"
)

# Load model once and cache it
# @st.cache_resource means it only loads once
# not every time someone uses the app
@st.cache_resource
def load_model():
    return YOLO('runs/detect/construction_safety/weights/best.pt')

model = load_model()

# APP HEADER


st.title("🦺 Construction Site Safety Detector")
st.markdown("""
This app uses a **custom trained YOLOv8 model** to detect safety violations
on construction sites in real time.

It can detect: `Hardhat` `NO-Hardhat` `Safety Vest` `NO-Safety Vest`
`Person` `Safety Cone` `Machinery` `Vehicle`
""")

st.divider()


# SIDEBAR SETTINGS


st.sidebar.title("⚙️ Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Only show detections above this confidence level"
)

mode = st.sidebar.radio(
    "Detection Mode",
    ["📷 Upload Image", "🎥 Live Webcam"]
)

st.sidebar.divider()
st.sidebar.markdown("""
### 🧠 About This Model
- **Architecture:** YOLOv8 Nano
- **Training:** 20 epochs on CPU
- **Dataset:** 717 construction site images
- **Classes:** 25 safety-related objects
- **Built with:** Python, NumPy, Ultralytics
""")

# MODE 1 — IMAGE UPLOAD


if mode == "📷 Upload Image":

    st.subheader("📷 Upload an Image")
    st.markdown("Upload any construction site photo to detect safety violations")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a construction site image"
    )      

    if uploaded_file is not None:

        # Convert uploaded file to image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Show original and result side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        # Run detection
        with st.spinner("Detecting safety violations..."):
            results = model(img_array, conf=confidence)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col2:
            st.markdown("**Detection Results**")
            st.image(annotated_rgb, use_column_width=True)

        # Show detection summary
        st.divider()
        st.subheader("📊 Detection Summary")

        boxes = results[0].boxes
        if len(boxes) == 0:
            st.info("No objects detected. Try lowering the confidence threshold.")
        else:
            # Count each class
            class_counts = {}
            violations    = []
            safe_items    = []

            for box in boxes:
                class_id   = int(box.cls[0])
                class_name = model.names[class_id]
                confidence_score = float(box.conf[0])

                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1

                if 'NO-' in class_name:
                    violations.append(f"{class_name} ({confidence_score:.0%})")
                else:
                    safe_items.append(f"{class_name} ({confidence_score:.0%})")

            # Show violations in red
            if violations:
                st.error(f"⚠️ Safety Violations Found: {len(violations)}")
                for v in violations:
                    st.markdown(f"- 🔴 {v}")

            # Show safe detections in green
            if safe_items:
                st.success(f"✅ Safe Items Detected: {len(safe_items)}")
                for s in safe_items:
                    st.markdown(f"- 🟢 {s}")

            # Show counts table
            st.subheader("📈 Detection Counts")
            count_col1, count_col2 = st.columns(2)
            items = list(class_counts.items())
            half  = len(items) // 2

            with count_col1:
                for name, count in items[:half]:
                    st.metric(name, count)

            with count_col2:
                for name, count in items[half:]:
                    st.metric(name, count)


# MODE 2 — LIVE WEBCAM

elif mode == "🎥 Live Webcam":

    st.subheader("🎥 Live Webcam Detection")
    st.markdown("Click **Start** to begin live safety detection from your webcam")

    run = st.toggle("▶️ Start Webcam")

    FRAME_WINDOW = st.image([])
    status_text  = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while run:
            success, frame = cap.read()

            if not success:
                st.error("Could not access webcam")
                break

            # Run detection
            results  = model(frame, conf=confidence, verbose=False)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Show frame in browser
            FRAME_WINDOW.image(annotated_rgb, use_container_width=True)

            # Show live detection status
            boxes = results[0].boxes
            if len(boxes) > 0:
                detections = []
                for box in boxes:
                    class_id   = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf_score = float(box.conf[0])
                    detections.append(f"{class_name} ({conf_score:.0%})")

                has_violation = any('NO-' in d for d in detections)

                if has_violation:
                    status_text.error(f"⚠️ VIOLATION: {', '.join(detections)}")
                else:
                    status_text.success(f"✅ SAFE: {', '.join(detections)}")
            else:
                status_text.info("👀 Scanning...")

        cap.release()
    else:
        st.info("Toggle the switch above to start your webcam")