import cv2
import numpy as np
from ultralytics import YOLO

FIXED_SIZE = (500, 500)

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

# --- Main Detection Function ---
def image_detection(model_path, image_path, brightness, zoom_percent, ROI_x, ROI_y, ROI_w, ROI_h):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Image not found!")
    
    image = image[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]
    
    # Adjust brightness
    bright_img = cv2.convertScaleAbs(image, alpha=1, beta=brightness)

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
    image = final_img
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    labeled_image = image_rgb.copy()

    # Load YOLO model
    model = YOLO(model_path)

    # Run detection
    detection_results = model(image_rgb, verbose=False)
    raw_boxes = []

    for result in detection_results[0].boxes.data.tolist():
        if len(result) < 6:
            continue
        x1, y1, x2, y2, conf, cls = map(float, result[:6])
        cls = int(cls)
        label = model.names.get(cls, "Unknown")
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
        cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(labeled_image, screw_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        print(screw_label)

    return labeled_image, summary["total_screws"], summary["normal"], summary["rust"], summary["chipped"], summary["bent"]
