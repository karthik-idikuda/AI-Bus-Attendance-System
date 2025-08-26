import torch
import cv2

# ----------------------------
# Load YOLOv5 mask detection model
# ----------------------------
def load_mask_model(model_path="models/mask_detection_yolov5s.pt", device='cpu'):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        model.to(device)
        print("[INFO] Mask detection YOLOv5 model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv5 model: {e}")
        return None

# ----------------------------
# Function to detect mask status in an image
# Returns: list of dicts with class name and confidence
# ----------------------------
def detect_mask(model, frame, conf_threshold=0.5):
    if model is None:
        return []

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    detections = []

    # Parse results
    for *box, conf, cls in results.xyxy[0]:
        if conf >= conf_threshold:
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "label": label,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2]
            })
    return detections

# ----------------------------
# Function to draw detections on frame
# ----------------------------
def draw_mask_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        confidence = det["confidence"]

        color = (0,255,0) if "mask" in label.lower() else (0,0,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# ----------------------------
# Example usage (testing)
# ----------------------------
if __name__ == "__main__":
    # Load model
    model = load_mask_model()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect masks
        detections = detect_mask(model, frame)
        # Draw results
        frame = draw_mask_detections(frame, detections)

        cv2.imshow("Mask Detection - YOLOv5", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
