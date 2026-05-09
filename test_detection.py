import cv2
from ultralytics import YOLO

#Test Pre-trained Object Detection


# Load the pre-trained YOLOv8 model
# 'yolov8n.pt' means YOLOv8 NANO - the smallest fastest version
# Perfect for CPU laptops - n = nano, s = small, m = medium
model = YOLO('yolov8s.pt')

# Open your webcam
# 0 means the first webcam connected to your computer
cap = cv2.VideoCapture(0)

print("Camera started! Press Q to quit.")

while True:

    # Read one frame from the webcam
    success, frame = cap.read()

    # If camera failed to read a frame, skip it
    if not success:
        print("Failed to read from camera")
        break

    # Run YOLO detection on this frame
    results = model(frame, verbose=False)

    # Draw the detection boxes on the frame
    annotated_frame = results[0].plot()

    # Show the frame in a window
    cv2.imshow("YOLOv8 Detection - Press Q to quit", annotated_frame)

    # If user presses Q, stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up - release camera and close windows
cap.release()
cv2.destroyAllWindows()
print("Camera stopped.")