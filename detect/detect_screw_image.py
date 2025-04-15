import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # --- Load Image ---
    image_path = ""
    output_path = ""

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        exit()

    # --- Load YOLO Model ---
    model_path = "best.pt"
    model = YOLO(model_path)

    # --- YOLO Detection ---
    detection_results = model(image, verbose=False)

    for result in detection_results[0].boxes.data.tolist():
        if len(result) < 6:
            continue
        x1, y1, x2, y2, conf, cls = map(float, result[:6])
        cls = int(cls)
        label = model.names.get(cls, "Unknown")

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # --- Save Output ---
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")
