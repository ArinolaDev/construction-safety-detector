from ultralytics import YOLO

#Train YOLOv8 On Custom Dataset

# Load the pre-trained nano model as our starting point
# We are NOT training from zero - we transfer its knowledge
model = YOLO('yolov8n.pt')

# Train on our construction safety dataset
results = model.train(
    data="Construction Site Safety.v1-original_raw-images.yolov8/data.yaml",
    epochs=20,
    imgsz=416,
    batch=4,
    device='cpu',
    workers=2,
    name='construction_safety',
    verbose=True
)

print("Training complete!")
print(f"Best model saved at: runs/detect/construction_safety/weights/best.pt")