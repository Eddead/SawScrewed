import streamlit as st
import cv2
import os
from detect_image import image_detection
from cam_detection import list_available_cameras
from ultralytics import YOLO

# Streamlit app layout
st.title("Screw Detection App")

# Select model
# Find all .pt files under subdirectories
models = []
def loop_through_folders(root_dir):
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path):
            # Recursively check inside subfolders
            loop_through_folders(entry_path)
        elif entry.endswith('.pt'):
            models.append(entry_path)
# Call the function
root_dir = '..\model_used'
loop_through_folders(root_dir)

# Streamlit selectbox to choose 
model = st.selectbox("Select a Model", models)

# # Select mode: Image, Video, Live
mode = st.radio("Choose the input type:", ("Image", "Video", "Live"))

if mode == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Save uploaded image locally
        image_path = os.path.join("uploaded_image.png")
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load original image for display
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Detect screw
        image_with_screw = image_detection(model, image_path)
        # Display original and detected images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(image_with_screw, caption="Image with Detected Screw", use_container_width=True)
            save_image = st.button("Save Image?")
            if save_image:
                cv2.imwrite("saved_detection.png", image_with_screw)
                st.success("Image saved successfully as saved_detection.png!")


elif mode == "Video":
    save_recording = st.checkbox("Save Detected Video?")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Save uploaded video
        video_path = os.path.join("uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Load YOLO model
        model_path = model
        model_using = YOLO(model_path)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video.")
        else:
            stframe = st.empty()
            # Prepare video writer if recording
            writer = None
            if save_recording:
                # Define codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 file
                fps = 20.0  # you can adjust this based on your camera
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter('ScrewDetectedVideo.mp4', fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # --- YOLO Inference ---
                results = model_using(frame, verbose=False)

                for result in results[0].boxes.data.tolist():
                    if len(result) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = map(float, result[:6])
                    cls = int(cls)
                    label = model_using.names.get(cls, "Unknown")
                    if conf < 0.3:
                        continue  # Skip everything else

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Write frame to file if recording
                if save_recording and writer is not None:
                    writer.write(frame)

                # Convert to RGB and show
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

            cap.release()
            if writer is not None:
                writer.release()
                st.success("Video saved successfully as ScrewDetectedVideo.mp4!")


elif mode == "Live":
    st.info("Camera access is required for live detection.")

    # Check available cameras
    available_cameras = list_available_cameras()
    available_options = available_cameras + ["IP Camera URL"]

    # Select Camera Source
    camera_source = st.selectbox("Select camera source:", available_options)

    custom_url = ""
    if camera_source == "IP Camera URL":
        custom_url = st.text_input("Enter IP camera URL (e.g., rtsp://...)")

    start_detection = st.button("Start Live Detection")
    start_recording = st.checkbox("Record Live View?")

    if start_detection:
        # Load model
        model_path = model
        model_using = YOLO(model_path)

        # Determine source
        if camera_source == "Webcam 0":
            source = 0
        elif camera_source == "Webcam 1":
            source = 1
        else:
            source = custom_url

        # Open camera
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.error(f"Failed to open camera source: {source}")
        else:
            stframe = st.empty()
            stop_detection = st.button("Stop Live Detection")

            # Prepare video writer if recording
            writer = None
            if start_recording:
                # Define codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 file
                fps = 20.0  # you can adjust this based on your camera
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter('recordedlog.mp4', fourcc, fps, (width, height))

            if stop_detection:
                cap.release()
                if writer is not None:
                    writer.release()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera disconnected or no frame captured.")
                    break

                # --- YOLO Inference ---
                results = model_using(frame, verbose=False)

                for result in results[0].boxes.data.tolist():
                    if len(result) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = map(float, result[:6])
                    cls = int(cls)
                    label = model_using.names.get(cls, "Unknown")
                    if conf < 0.3:
                        continue

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Write frame to file if recording
                if start_recording and writer is not None:
                    writer.write(frame)

                # Convert to RGB and show
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

            cap.release()
            if writer is not None:
                writer.release()
    