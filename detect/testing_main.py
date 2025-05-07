import streamlit as st
import cv2
import os
import numpy as np
import tempfile
from detect_image import image_detection, resize_and_pads
from cam_detection import list_available_cameras
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("🔩 Screw Detection App")

FIXED_SIZE = (500, 500)

models = []
def loop_through_folders(root_dir):
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path):
            loop_through_folders(entry_path)
        elif entry.endswith('.pt'):
            models.append(entry_path)
            
# --- Compute IoU between two boxes ---
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    xa1, ya1, xa2, ya2 = box2

    xi1 = max(x1, xa1)
    yi1 = max(y1, ya1)
    xi2 = min(x2, xa2)
    yi2 = min(y2, ya2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xa2 - xa1) * (ya2 - ya1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# --- Merge boxes by taking min/max of coordinates ---
def merge_boxes(group):
    x1_list = [box[0] for _, box in group]
    y1_list = [box[1] for _, box in group]
    x2_list = [box[2] for _, box in group]
    y2_list = [box[3] for _, box in group]
    return [min(x1_list), min(y1_list), max(x2_list), max(y2_list)]

loop_through_folders('../model_used')

model = st.selectbox("📦 Select a Model", models)
mode = st.radio("📷 Choose Input Type", ("Image", "Video", "Live"))

def apply_zoom(frame, zoom_factor):
    if zoom_factor == 1.0:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    x1, y1 = (w - new_w) // 2, (h - new_h) // 2
    x2, y2 = x1 + new_w, y1 + new_h
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))

# ========== IMAGE MODE ==========
if mode == "Image":
    uploaded_file = st.file_uploader("🖼️ Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        # Save uploaded image
        image_path = os.path.join("uploaded_image.png")
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        roi_used_image = resize_and_pad(image_rgb)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_path = temp_file.name  # Path to the temp file
        temp_file.close()  # Close file handle

        # Save roi selected image to temp path
        cv2.imwrite(temp_path, roi_used_image)

        # Ask ROI only when a new file is uploaded
        if "last_uploaded" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded:
            roi = cv2.selectROI("Select ROI", roi_used_image)
            cv2.destroyAllWindows()

            if roi == (0, 0, 0, 0):
                st.warning("❗ Please select a valid ROI!")
                st.stop()

            st.session_state.roi = roi
            st.session_state.last_uploaded = uploaded_file.name
        
        x, y, w, h = st.session_state.roi
        cropped = roi_used_image[y:y+h, x:x+w]

        brightness = st.slider('✨ Brightness', -100, 100, 0)
        zoom_percent = st.slider('🔍 Zoom %', 10, 300, 100)
        zoom = zoom_percent / 100.0

        bright_img = cv2.convertScaleAbs(cropped, alpha=1, beta=brightness)
        zoomed_img = cv2.resize(bright_img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

        zh, zw = zoomed_img.shape[:2]
        ch, cw = FIXED_SIZE[1], FIXED_SIZE[0]
        center_y, center_x = zh // 2, zw // 2
        y1 = max(center_y - ch // 2, 0)
        y2 = min(y1 + ch, zh)
        x1 = max(center_x - cw // 2, 0)
        x2 = min(x1 + cw, zw)
        cropped_img = zoomed_img[y1:y2, x1:x2]

        final_img = np.ones((ch, cw, 3), dtype=np.uint8) * 255
        cropped_h, cropped_w = cropped_img.shape[:2]
        y_offset = (ch - cropped_h) // 2
        x_offset = (cw - cropped_w) // 2
        final_img[y_offset:y_offset+cropped_h, x_offset:x_offset+cropped_w] = cropped_img

        detected_img, total, normal, rust, chipped, bent = image_detection(
            model, temp_path, brightness, zoom_percent, x, y, w, h
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(final_img, caption="🖼️ ROI Image")
        with col2:
            st.image(detected_img, caption="✅ Detected Screws")

        if st.button("💾 Save Detection"):
            cv2.imwrite("saved_detection.png", detected_img)
            st.success("Image saved as `saved_detection.png`!")

        st.markdown("### 📊 Detection Summary")
        st.markdown(f"- **Total Screws**: {total}")
        st.markdown(f"- 🟢 Normal: {normal}")
        st.markdown(f"- 🟠 Rust: {rust}")
        st.markdown(f"- 🔵 Bent: {bent}")
        st.markdown(f"- 🟣 Chipped: {chipped}")

# ========== VIDEO MODE ==========
elif mode == "Video":
    uploaded_video = st.file_uploader("🎥 Upload a video...", type=["mp4", "avi", "mov"])
    if uploaded_video:
        brightness = st.slider('✨ Brightness', -100, 100, 0)
        zoom_factor = st.slider('🔍 Zoom Factor', 1.0, 3.0, 1.0, step=0.1)

        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st_video_preview = st.empty()

        if "previewing" not in st.session_state:
            st.session_state.previewing = False
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.previewing and st.button("👁️ Preview Video"):
                st.session_state.previewing = True
                st.experimental_rerun()  # Trigger a rerun
        with col2:
            if st.session_state.previewing and st.button("🛑 Stop Preview Video"):
                st.session_state.previewing = False
                st.experimental_rerun()  # Trigger a rerun
        
                    
        if st.session_state.previewing:
            cap = cv2.VideoCapture(video_path)
            st_video_preview = st.empty()
            st.info("🎞️ Showing preview with brightness & zoom adjustments...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or not st.session_state.previewing:
                    break
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
                frame = apply_zoom(frame, zoom_factor)
                st_video_preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            cap.release()
            st.success("✅ Preview stopped.")
            st.session_state.previewing = False
                
        if st.button("🚀 Start Detection"):
            model_using = YOLO(model)
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            w, h = int(cap.get(3)), int(cap.get(4))
            writer = cv2.VideoWriter("ScrewDetectedVideo.mp4", fourcc, fps, (w, h))

            st.info("🔍 Detecting...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
                frame = apply_zoom(frame, zoom_factor)
                results = model_using(frame, verbose=False)

                raw_boxes = []

                for result in results[0].boxes.data.tolist():
                    if len(result) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = map(float, result[:6])
                    cls = int(cls)
                    label = model_using .names.get(cls, "Unknown")
                    raw_boxes.append((label, [x1, y1, x2, y2]))

                # Group overlapping boxes
                groups = []
                used = set()

                for i, (label1, box1) in enumerate(raw_boxes):
                    if i in used:
                        continue
                    group = [(label1, box1)]
                    used.add(i)
                    for j in range(i + 1, len(raw_boxes)):
                        if j in used:
                            continue
                        label2, box2 = raw_boxes[j]
                        if iou(box1, box2) >= 0.6:
                            group.append((label2, box2))
                            used.add(j)
                    groups.append(group)

                # Count summary
                summary = {
                    "total_screws": 0,
                    "normal": 0,
                    "chipped": 0,
                    "rust": 0,
                    "bent": 0
                }

                # Label and draw each screw group
                for idx, group in enumerate(groups, start=1):
                    summary["total_screws"] += 1
                    labels = set(label for label, _ in group)

                    # If there's any anomaly, we remove 'normal'
                    anomalies = {"chipped", "rust", "bent"}
                    detected_anomalies = labels & anomalies

                    if detected_anomalies:
                        if "chipped" in detected_anomalies:
                            summary["chipped"] += 1
                        if "rust" in detected_anomalies:
                            summary["rust"] += 1
                        if "bent" in detected_anomalies:
                            summary["bent"] += 1
                        labels -= {"normal"}  # Exclude 'normal' from label string if anomaly exists
                    else:
                        summary["normal"] += 1

                    merged_box = merge_boxes(group)
                    x1, y1, x2, y2 = map(int, merged_box)
                    combined_label = ", ".join(sorted(labels))
                    screw_label = f"Screw #{idx}: {combined_label}"

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, screw_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                summaryText = f"Total: {summary["total_screws"]}  Normal: {summary["normal"]}  Rust: {summary["rust"]}  Bent: {summary["bent"]}  Chipped: {summary["chipped"]}"
                cv2.putText(frame, summaryText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                writer.write(frame)

            cap.release()
            writer.release()
            st.success("✅ Video saved as `ScrewDetectedVideo.mp4`!")

# ========== LIVE CAMERA MODE ==========
elif mode == "Live":
    st.info("📷 Camera access required for live detection.")
    available_cameras = list_available_cameras()
    camera_source = st.selectbox("Choose camera source:", available_cameras + ["IP Camera URL"])

    custom_url = ""
    if camera_source == "IP Camera URL":
        custom_url = st.text_input("Enter IP camera URL (e.g., rtsp://...)")

    brightness = st.slider('✨ Brightness', -100, 100, 0)
    zoom_factor = st.slider('🔍 Zoom Factor', 1.0, 3.0, 1.0, step=0.1)

    # Initialize session state
    if "live_detection" not in st.session_state:
        st.session_state.live_detection = False
    if "live_saved" not in st.session_state:
        st.session_state.live_saved = False
    if "writer" not in st.session_state:
        st.session_state.writer = None

    source = 0 if camera_source == "Webcam 0" else 1 if camera_source == "Webcam 1" else custom_url
    model_using = YOLO(model)
    stframe = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.live_detection and st.button("🚀 Start Live Detection"):
            st.session_state.live_detection = True
            st.session_state.live_saved = False
            st.experimental_rerun()

    with col2:
        if st.session_state.live_detection and st.button("🛑 Stop Live Detection"):
            st.session_state.live_detection = False
            if st.session_state.writer:
                st.session_state.writer.release()
                st.session_state.writer = None
                st.session_state.live_saved = True
            st.experimental_rerun()

    if st.session_state.live_detection:
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            st.error("❌ Failed to open camera.")
            st.session_state.live_detection = False
        else:
            st.info("🎥 Live detection running...")
            try:
                while cap.isOpened() and st.session_state.live_detection:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Initialize video writer if not already created
                    if st.session_state.writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        st.session_state.writer = cv2.VideoWriter("LiveScrewDetection.mp4", fourcc, 20.0, (frame.shape[1], frame.shape[0]))

                    # Adjust brightness and zoom
                    frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
                    frame = apply_zoom(frame, zoom_factor)

                    # Run detection
                    results = model_using(frame, verbose=False)
                    raw_boxes = []
                    for result in results[0].boxes.data.tolist():
                        if len(result) < 6:
                            continue
                        x1, y1, x2, y2, conf, cls = map(float, result[:6])
                        label = model_using.names.get(int(cls), "Unknown")
                        raw_boxes.append((label, [x1, y1, x2, y2]))

                    # Group boxes and summarize
                    groups = []
                    used = set()
                    for i, (label1, box1) in enumerate(raw_boxes):
                        if i in used:
                            continue
                        group = [(label1, box1)]
                        used.add(i)
                        for j in range(i + 1, len(raw_boxes)):
                            if j in used:
                                continue
                            label2, box2 = raw_boxes[j]
                            if iou(box1, box2) >= 0.6:
                                group.append((label2, box2))
                                used.add(j)
                        groups.append(group)

                    summary = {"total_screws": 0, "normal": 0, "chipped": 0, "rust": 0, "bent": 0}
                    for idx, group in enumerate(groups, start=1):
                        summary["total_screws"] += 1
                        labels = set(label for label, _ in group)
                        anomalies = {"chipped", "rust", "bent"}
                        detected = labels & anomalies
                        if detected:
                            for a in detected:
                                summary[a] += 1
                            labels.discard("normal")
                        else:
                            summary["normal"] += 1

                        x1, y1, x2, y2 = map(int, merge_boxes(group))
                        label_text = ", ".join(sorted(labels))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"Screw #{idx}: {label_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Show summary on top
                    summary_text = f"Total: {summary['total_screws']}  Normal: {summary['normal']}  Rust: {summary['rust']}  Bent: {summary['bent']}  Chipped: {summary['chipped']}"
                    cv2.putText(frame, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if st.session_state.writer:
                        st.session_state.writer.write(frame)
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            finally:
                cap.release()
                if st.session_state.writer:
                    st.session_state.writer.release()
                    st.session_state.writer = None
                    st.session_state.live_saved = True
                st.session_state.live_detection = False
                st.experimental_rerun()
    
    if st.session_state.live_saved:
        st.success("💾 Recording saved as LiveScrewDetection.mp4.")
