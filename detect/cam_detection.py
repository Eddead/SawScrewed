import cv2

def list_available_cameras():
    cv2.setLogLevel(0)  # Silence OpenCV error messages
    available_cameras = []
    
    cap = cv2.VideoCapture(0)
    if cap is not None and cap.isOpened():
        available_cameras.append("Webcam 0")
        cap.release()

    return available_cameras