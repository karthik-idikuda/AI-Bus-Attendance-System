import os
import pickle
import cv2

# ----------------------------
# Function to create directories if not exist
# ----------------------------
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created folder: {path}")

# ----------------------------
# Function to save pickle file
# ----------------------------
def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[INFO] Saved pickle file: {file_path}")

# ----------------------------
# Function to load pickle file
# ----------------------------
def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"[INFO] Loaded pickle file: {file_path}")
        return data
    else:
        print(f"[WARN] Pickle file not found: {file_path}")
        return None

# ----------------------------
# Function to capture single image from webcam
# Returns captured frame
# ----------------------------
def capture_image_from_cam(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera.")
        return None

    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    else:
        print("[ERROR] Failed to capture image.")
        return None

# ----------------------------
# Function to resize image with aspect ratio
# ----------------------------
def resize_image(img, width=None, height=None):
    if width is None and height is None:
        return img

    h, w = img.shape[:2]

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

# ----------------------------
# Example usage (testing)
# ----------------------------
if __name__ == "__main__":
    # Test folder creation
    create_folder("test_dir")

    # Test pickle save/load
    test_data = {"name": "John Doe", "id": "S123"}
    save_pickle(test_data, "test_dir/test.pkl")
    loaded = load_pickle("test_dir/test.pkl")
    print("Loaded Data:", loaded)

    # Test image capture
    img = capture_image_from_cam()
    if img is not None:
        cv2.imshow("Captured Image", resize_image(img, width=640))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
