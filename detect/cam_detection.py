import cv2

def list_available_cameras(max_cameras=2):
    cv2.setLogLevel(0)  # Silence OpenCV error messages
    available_cameras = []
    for i in range (max_cameras):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available_cameras.append("Webcam "+str(i))
            cap.release()

    return available_cameras
