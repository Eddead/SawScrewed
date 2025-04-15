import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # Load YOLO model
    model_path = "best.pt"
    model = YOLO(model_path)

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, verbose=False)

        for result in results[0].boxes.data.tolist():
            if len(result) < 6:
                continue
            x1, y1, x2, y2, conf, cls = map(float, result[:6])
            cls = int(cls)
            label = model.names.get(cls, "Unknown")

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display and save
        cv2.imshow("YOLO Webcam Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
