import cv2
from ultralytics import YOLO

def image_detection(model_path,image_path):
    # --- Load Image ---
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise ValueError("Error: Image not found!")

    # --- Load YOLO Model ---
    model = YOLO(model_path)

    # --- YOLO Detection ---
    detection_results = model(image, verbose=False)

    for result in detection_results[0].boxes.data.tolist():
        if len(result) < 6:
            continue
        x1, y1, x2, y2, conf, cls = map(float, result[:6])
        cls = int(cls)

        try:
            label = model.names[cls]
        except (IndexError, KeyError):
            label = "Unknown"

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # --- Return the image with drawings ---
    return image
