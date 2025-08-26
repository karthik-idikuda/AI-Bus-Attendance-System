import cv2
import mediapipe as mp

# ----------------------------
# Initialize Mediapipe Face Detection
# ----------------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ----------------------------
# Function to initialize video capture
# ----------------------------
def initialize_camera(camera_index=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

# ----------------------------
# Function to detect faces in a frame
# Returns list of bounding boxes [x, y, w, h]
# ----------------------------
def detect_faces(frame, confidence_threshold=0.5):
    results_boxes = []
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=confidence_threshold) as face_detection:
        
        # Convert to RGB as Mediapipe requires
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)

                # Ensure box within frame bounds
                x = max(0, x)
                y = max(0, y)
                results_boxes.append([x, y, width, height])

    return results_boxes

# ----------------------------
# Function to draw bounding boxes on frame
# ----------------------------
def draw_boxes(frame, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

# ----------------------------
# Example usage (testing)
# ----------------------------
if __name__ == "__main__":
    cap = initialize_camera()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(frame)
        frame = draw_boxes(frame, boxes)

        cv2.imshow("Face Detection - Mediapipe", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
