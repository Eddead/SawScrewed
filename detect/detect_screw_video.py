if __name__ == '__main__':
    import cv2
    # Using YOLO
    import os
    from ultralytics import YOLO

    video_path = ""
    output_path = ""

    # --- Load Video ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    # --- Setup video writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Load YOLO Model ---
    model_path = "best.pt"
    model = YOLO(model_path)
    
    # --- Frame Counter ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_past_percentage = int(frame_count/total_frames*100)
        print(f"Processing frame: {frame_count}/{total_frames} {frame_past_percentage}%", end="\r")

        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        # height, width, _ = frame.shape
        # new_width = (width - height) // 2
        # frame = frame[:, new_width:height + new_width]

        # --- YOLO Detection ---
        detection_results = model(frame, verbose=False)

        for result in detection_results[0].boxes.data.tolist():  # Extract detection data
            if len(result) < 6:
                continue
            x1, y1, x2, y2, conf, cls = map(float, result[:6])
            cls = int(cls)
            label = model.names.get(cls, "Unknown")

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Write to file
        out.write(frame)

        # --- Show Frame (Optional) ---
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        

    cap.release()
    out.release()
    cv2.destroyAllWindows()
