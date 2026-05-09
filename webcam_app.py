import cv2
from ultralytics import YOLO

# PIECE 5: Live Webcam App With YOUR Trained Model

# Load YOUR custom trained model - not the generic one
model = YOLO('runs/detect/construction_safety/weights/best.pt')

# Define colours for each detection box
COLOURS = {
    'Hardhat'       : (0, 255, 0),    # Green
    'NO-Hardhat'    : (0, 0, 255),    # Red
    'Safety Vest'   : (0, 255, 255),  # Yellow
    'NO-Safety Vest': (0, 0, 255),    # Red
    'Person'        : (255, 165, 0),  # Orange
    'Safety Cone'   : (255, 0, 255),  # Purple
}

# Open webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Safety Detection App Running!")
print("Press Q to quit")
print("-" * 40)

while True:

    success, frame = cap.read()

    if not success:
        print("Camera error")
        break

    # Run detection with confidence threshold
    # Only show detections above 30% confidence
    results = model(frame, conf=0.3, verbose=False)

    # Count what was detected this frame
    detections = results[0].boxes
    detected_objects = []

    for box in detections:
        # Get the class name of this detection
        class_id   = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        detected_objects.append(f"{class_name} ({confidence:.0%})")

    # Draw detection boxes on frame
    annotated_frame = results[0].plot()

    # Add title bar at top of frame
    cv2.rectangle(annotated_frame, (0, 0), (640, 40), (0, 0, 0), -1)
    cv2.putText(
        annotated_frame,
        "Construction Safety Detector | Press Q to quit",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )

    # Print detections to terminal
    if detected_objects:
        print(f"Detected: {', '.join(detected_objects)}")

    # Show the frame
    cv2.imshow("Construction Safety Detector", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("App closed.")