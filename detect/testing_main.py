import streamlit as st
import cv2
import os
import numpy as np
from test_detect import image_detection
from cam_detection import list_available_cameras
from ultralytics import YOLO

# Streamlit app layout
st.title("Screw Detection App")

FIXED_SIZE = (500, 500)
# Count Object
count = 0
countStatus = [0,0]

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

def apply_zoom(frame, zoom_factor):
    if zoom_factor == 1.0:
        return frame  # No zoom

    height, width = frame.shape[:2]
    # Calculate cropping size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    # Calculate cropping margins
    x1 = (width - new_width) // 2
    y1 = (height - new_height) // 2
    x2 = x1 + new_width
    y2 = y1 + new_height

    # Crop and resize back to original size
    cropped = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(cropped, (width, height))

    return zoomed_frame


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
        
        # Sliders
        brightness = st.slider('Brightness', -100, 100, 0)
        zoom_percent = st.slider('Zoom %', 10, 300, 100)

        # Adjust brightness
        bright_img = cv2.convertScaleAbs(original_image, alpha=1, beta=brightness)

        # Adjust zoom
        zoom = zoom_percent / 100.0
        zoomed_img = cv2.resize(bright_img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

        # Get center crop
        zh, zw = zoomed_img.shape[:2]
        ch, cw = FIXED_SIZE[1], FIXED_SIZE[0]

        center_y = zh // 2
        center_x = zw // 2

        # Calculate crop box
        y1 = max(center_y - ch // 2, 0)
        y2 = y1 + ch
        x1 = max(center_x - cw // 2, 0)
        x2 = x1 + cw

        # Make sure we don't go out of bounds
        y2 = min(y2, zh)
        y1 = max(y2 - ch, 0)
        x2 = min(x2, zw)
        x1 = max(x2 - cw, 0)

        cropped_img = zoomed_img[y1:y2, x1:x2]

        # If image is smaller than frame, pad with white
        final_img = np.ones((ch, cw, 3), dtype=np.uint8) * 255
        cropped_h, cropped_w = cropped_img.shape[:2]
        y_offset = (ch - cropped_h) // 2
        x_offset = (cw - cropped_w) // 2
        final_img[y_offset:y_offset+cropped_h, x_offset:x_offset+cropped_w] = cropped_img

        # Detect screw
        image_with_screw, count, countStatus = image_detection(model, image_path, brightness, zoom_percent)
        
        # Display original and detected images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(final_img, caption="Original Image")
        with col2:
            st.image(image_with_screw, caption="Image with Detected Screw")
        
        col3, col4, col5 = st.columns([1, 2, 1])
        with col4:
            save_image = st.button("Save Image?")
            if save_image:
                cv2.imwrite("saved_detection.png", image_with_screw)
                st.success("Image saved successfully as saved_detection.png!")
            
        st.text(f"Total Screws: {count}")
        st.text(f"Normal Screws: {countStatus[0]}")
        st.text(f"Rusty Screws: {countStatus[1]}")

elif mode == "Video":
    save_recording = st.checkbox("Save Detected Video?")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Sliders
        brightness = st.slider('Brightness', -100, 100, 0)
        zoom_factor = st.slider('Zoom', 1.0, 3.0, 1.0, step=0.1)

        # Save uploaded video temporarily
        video_path = os.path.join("uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Open video for preview
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        preview = st.button("Preview Video with Brightness")
        detect = st.button("Start Detection")

        if preview and not detect:
            st.text("Preview:")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
                frame = apply_zoom(frame, zoom_factor)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")
            cap.release()

        if detect:
            st.text("Detect:")
            cap = cv2.VideoCapture(video_path)  # Re-open because previous cap is finished
            model_using = YOLO(model)

            if save_recording:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 20.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter('ScrewDetectedVideo.mp4', fourcc, fps, (width, height))
            else:
                writer = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
                frame = apply_zoom(frame, zoom_factor)
                results = model_using(frame, verbose=False)

                # --- (same detection code as yours) ---
                count = 0
                countStatus = [0, 0]
                for result in results[0].boxes.data.tolist():
                    if len(result) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = map(float, result[:6])
                    cls = int(cls)
                    label = model_using.names.get(cls, "Unknown")
                    if conf < 0.3:
                        continue

                    if label == 'normal':
                        if conf < 0.7:
                            continue
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if label == 'normal':
                        countStatus[0] += 1
                    else:
                        countStatus[1] += 1
                    count += 1

                cv2.putText(frame, f"Total Screws: {count}", (10, frame.shape[0] - 10 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Normal Screws: {countStatus[0]}", (10, frame.shape[0] - 10 - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Rusty Screws: {countStatus[1]}", (10, frame.shape[0] - 10 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                if save_recording and writer is not None:
                    writer.write(frame)

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
    # Sliders
    brightness = st.slider('Brightness', -100, 100, 0)
    zoom_factor = st.slider('Zoom', 1.0, 3.0, 1.0, step=0.1)
    
    #buttons
    start_detection = st.button("Start Live Detection")
    test_camera = st.button("Test Camera")
    start_recording = st.checkbox("Record Live View?")
    
    if test_camera and not start_detection:
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

            if stop_detection:
                cap.release()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera disconnected or no frame captured.")
                    break

                frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
                frame = apply_zoom(frame, zoom_factor)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

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
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
                frame = apply_zoom(frame, zoom_factor)
                results = model_using(frame, verbose=False)

                # --- (same detection code as yours) ---
                count = 0
                countStatus = [0, 0]

                for result in results[0].boxes.data.tolist():
                    if len(result) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = map(float, result[:6])
                    cls = int(cls)
                    label = model_using.names.get(cls, "Unknown")
                    if conf < 0.3:
                        continue

                    if label == 'normal':
                        if conf < 0.7:
                            continue
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if label == 'normal':
                        countStatus[0] += 1
                    else:
                        countStatus[1] += 1
                    count += 1

                cv2.putText(frame, f"Total Screws: {count}", (10, frame.shape[0] - 10 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Normal Screws: {countStatus[0]}", (10, frame.shape[0] - 10 - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Rusty Screws: {countStatus[1]}", (10, frame.shape[0] - 10 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                # Write frame to file if recording
                if start_recording and writer is not None:
                    writer.write(frame)

                # Convert to RGB and show
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

            cap.release()
            if writer is not None:
                writer.release()
    
