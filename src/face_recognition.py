import cv2
import numpy as np
import pickle
import os
import time
from numpy import linalg as LA

# ----------------------------
# Simple Face Recognition (No TensorFlow)
# ----------------------------

def load_facenet_model(model_path="models/facenet_keras.h5"):
    """
    Fallback to simple feature extraction if TensorFlow model fails
    """
    try:
        # Try to import and load TensorFlow model
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print("[INFO] FaceNet model loaded successfully.")
        return model
    except Exception as e:
        print(f"[WARNING] TensorFlow model loading failed: {e}")
        print("[INFO] Falling back to simple feature extraction...")
        return "simple_features"

def preprocess_face(img):
    """Preprocess face image"""
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    if std > 0:
        img = (img - mean) / std
    img = np.expand_dims(img, axis=0)
    return img

def get_face_embedding(model, face_img):
    """
    Get face embedding - either from TensorFlow model or simple features
    """
    if isinstance(model, str) and model == "simple_features":
        # Simple feature extraction using histogram and basic stats
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray = cv2.resize(gray, (64, 64))
        
        # Extract features
        features = []
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        features.extend(hist.flatten())
        
        # Statistical features
        features.extend([
            gray.mean(),
            gray.std(),
            gray.min(),
            gray.max(),
            np.median(gray)
        ])
        
        # Texture features (simple)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(laplacian)
        
        # Convert to numpy array and normalize
        embedding = np.array(features, dtype=np.float32)
        
        # Normalize to unit vector
        norm = LA.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    else:
        # Use TensorFlow model
        processed_img = preprocess_face(face_img)
        embedding = model.predict(processed_img, verbose=0)
        return embedding[0]

def load_embeddings_db(embeddings_path="data/embeddings/embeddings.pkl"):
    """Load embeddings database"""
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            embeddings_db = pickle.load(f)
        print(f"[INFO] Embeddings database loaded with {len(embeddings_db)} entries.")
    else:
        embeddings_db = {}
        print("[INFO] No embeddings database found. Starting fresh.")
    return embeddings_db

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (LA.norm(a) * LA.norm(b))

def recognize_face(embedding, embeddings_db, threshold=0.6):
    """
    Recognize face by comparing embedding with database
    Adjusted for fallback feature extraction with more appropriate thresholds
    """
    if len(embeddings_db) == 0:
        return "Unknown", 0.0

    best_match_name = "Unknown"
    best_score = 0.0
    second_best_score = 0.0
    all_scores = []  # Track all similarity scores for distribution analysis

    for name, stored_embedding in embeddings_db.items():
        # Calculate similarity
        similarity = cosine_similarity(embedding, stored_embedding)
        all_scores.append(similarity)
        
        if similarity > best_score:
            second_best_score = best_score
            best_score = similarity
            best_match_name = name
        elif similarity > second_best_score:
            second_best_score = similarity

    print(f"[DEBUG] Recognition scores - Best: {best_score:.3f} ({best_match_name}), Second: {second_best_score:.3f}")
    print(f"[DEBUG] All scores: {[f'{score:.3f}' for score in all_scores]}")

    # Relaxed validation for fallback feature extraction:
    # 1. Primary threshold check - lowered to 0.6 for fallback features
    if best_score < threshold:
        print(f"[DEBUG] Rejected by threshold: {best_score:.3f} < {threshold}")
        return "Unknown", best_score
    
    # 2. Skip confidence gap check for very high confidence scores (>= 0.95)
    if best_score >= 0.95:
        print(f"[DEBUG] Match accepted: {best_match_name} with high confidence {best_score:.3f} (skipped gap check)")
        return best_match_name, best_score
    
    # 3. Confidence gap check - only apply for moderate confidence and multiple entries
    if len(embeddings_db) > 1 and best_score < 0.95:  # Only check gap for moderate confidence scores
        confidence_gap = best_score - second_best_score
        if confidence_gap < 0.02:  # Very small gap requirement
            print(f"[DEBUG] Rejected by confidence gap: {confidence_gap:.3f} < 0.02 (moderate confidence)")
            return "Unknown", best_score
    
    print(f"[DEBUG] Match accepted: {best_match_name} with score {best_score:.3f}")
    return best_match_name, best_score

def add_face_to_db(name, embedding, embeddings_db, save_path="data/embeddings/embeddings.pkl"):
    """Add new face to embeddings database"""
    embeddings_db[name] = embedding
    
    # Save updated database
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings_db, f)
    
    print(f"[INFO] Added {name} to embeddings database.")

def validate_face_quality(face_img, min_size=(50, 50), blur_threshold=100):
    """
    Validate face quality before processing
    Returns: (is_valid, quality_score)
    """
    if face_img is None or face_img.size == 0:
        return False, 0.0
    
    h, w = face_img.shape[:2]
    
    # Check minimum size
    if h < min_size[0] or w < min_size[1]:
        return False, 0.1
    
    # Check for blur using Laplacian variance
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Check brightness and contrast
    brightness = gray.mean()
    contrast = gray.std()
    
    # Quality scoring
    size_score = min(1.0, (h * w) / (100 * 100))  # Normalize by 100x100
    blur_score_norm = min(1.0, blur_score / blur_threshold)
    brightness_score = 1.0 - abs(brightness - 128) / 128  # Prefer balanced brightness
    contrast_score = min(1.0, contrast / 50)  # Good contrast
    
    quality_score = (size_score + blur_score_norm + brightness_score + contrast_score) / 4
    
    # Face is valid if it meets minimum quality thresholds
    is_valid = (
        blur_score > blur_threshold * 0.5 and  # Not too blurry
        brightness > 30 and brightness < 230 and  # Not too dark/bright
        contrast > 20 and  # Has some contrast
        quality_score > 0.4  # Overall quality threshold
    )
    
    return is_valid, quality_score

# ----------------------------
# Multi-face processing functions
# ----------------------------
def process_multiple_faces(model, frame, face_boxes, embeddings_db, recognition_threshold=0.75, 
                          min_quality_score=0.5, cooldown_time=5):
    """
    Process multiple faces in a frame and return recognition results
    Returns: List of tuples (name, confidence, face_box)
    """
    results = []
    
    # Dictionary to track last recognition time for each person
    # This should be maintained by the caller and passed in as an argument for a real implementation
    # Here we use a static variable as a simple demonstration
    if not hasattr(process_multiple_faces, "last_recognition_time"):
        process_multiple_faces.last_recognition_time = {}
    
    current_time = time.time()
    
    for box in face_boxes:
        x, y, w, h = box
        face_img = frame[y:y+h, x:x+w]
        
        # Validate face quality
        is_valid, quality_score = validate_face_quality(face_img)
        
        if is_valid and quality_score >= min_quality_score:
            # Get face embedding
            embedding = get_face_embedding(model, face_img)
            
            # Recognize face
            name, confidence = recognize_face(embedding, embeddings_db, threshold=recognition_threshold)
            
            # Apply cooldown to prevent duplicate recognitions
            if name != "Unknown":
                # Check if this person was recently recognized
                last_time = process_multiple_faces.last_recognition_time.get(name, 0)
                if current_time - last_time < cooldown_time:
                    # Skip this face, it was recently recognized
                    continue
                    
                # Update last recognition time
                process_multiple_faces.last_recognition_time[name] = current_time
            
            results.append((name, confidence, box))
    
    # Clean up old entries from last_recognition_time
    for name in list(process_multiple_faces.last_recognition_time.keys()):
        if current_time - process_multiple_faces.last_recognition_time[name] > cooldown_time * 2:
            del process_multiple_faces.last_recognition_time[name]
    
    return results

def get_unique_faces(recognition_results, confidence_threshold=0.75):
    """
    Filter recognition results to get unique names with high confidence
    Returns: List of unique names that passed the confidence threshold
    """
    unique_names = set()
    for name, confidence, _ in recognition_results:
        if name != "Unknown" and confidence >= confidence_threshold:
            unique_names.add(name)
    return list(unique_names)

# ----------------------------
# Example usage for multi-face attendance
# ----------------------------
def example_multi_face_attendance():
    """
    Example function showing how to use the multi-face recognition for attendance
    """
    from src.attendance import mark_attendance
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Load model and embeddings
    model = load_facenet_model()
    embeddings_db = load_embeddings_db()
    
    # Tracking variables
    processed_names = set()  # Track names that have been processed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
            
        # Detect faces
        face_boxes = []
        try:
            # Assuming you have a face detection function
            from src.face_detection import detect_faces
            face_boxes = detect_faces(frame)
        except Exception as e:
            print(f"[ERROR] Face detection error: {e}")
            
        # Process detected faces
        recognition_results = process_multiple_faces(
            model, frame, face_boxes, embeddings_db, 
            recognition_threshold=0.75,  # Higher threshold for stricter matching
            min_quality_score=0.5,       # Minimum quality score for face validation
            cooldown_time=5              # Time in seconds before same person can be recognized again
        )
        
        # Draw results on frame
        for name, confidence, box in recognition_results:
            x, y, w, h = box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display name and confidence
            text = f"{name} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Mark attendance for newly detected students
            if name != "Unknown" and name not in processed_names:
                # You would need to have student IDs for each name
                student_id = name  # Placeholder, in real system you would look up the ID
                mark_attendance(name, student_id)
                processed_names.add(name)
                print(f"Marked attendance for {name}")
        
        # Display the frame
        cv2.imshow("Multi-face Attendance", frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
# ----------------------------
# Test function
# ----------------------------
if __name__ == "__main__":
    print("Testing face recognition module...")
    
    # Test model loading
    model = load_facenet_model()
    print(f"Model type: {type(model)}")
    
    # Test embeddings database
    embeddings_db = load_embeddings_db()
    print(f"Embeddings database has {len(embeddings_db)} entries")
    
    print("Face recognition module test completed.")
    
    # Uncomment to test multi-face attendance system
    # print("Starting multi-face attendance test...")
    # example_multi_face_attendance()
    """
    Example function showing how to use the multi-face recognition for attendance
    """
    from src.attendance import mark_attendance
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Load model and embeddings
    model = load_facenet_model()
    embeddings_db = load_embeddings_db()
    
    # Tracking variables
    processed_names = set()  # Track names that have been processed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
            
        # Detect faces
        face_boxes = []
        try:
            # Assuming you have a face detection function
            from src.face_detection import detect_faces
            face_boxes = detect_faces(frame)
        except Exception as e:
            print(f"[ERROR] Face detection error: {e}")
            
        # Process detected faces
        recognition_results = process_multiple_faces(
            model, frame, face_boxes, embeddings_db, 
            recognition_threshold=0.75,  # Higher threshold for stricter matching
            min_quality_score=0.5,       # Minimum quality score for face validation
            cooldown_time=5              # Time in seconds before same person can be recognized again
        )
        
        # Draw results on frame
        for name, confidence, box in recognition_results:
            x, y, w, h = box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display name and confidence
            text = f"{name} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Mark attendance for newly detected students
            if name != "Unknown" and name not in processed_names:
                # You would need to have student IDs for each name
                student_id = name  # Placeholder, in real system you would look up the ID
                mark_attendance(name, student_id)
                processed_names.add(name)
                print(f"Marked attendance for {name}")
        
        # Display the frame
        cv2.imshow("Multi-face Attendance", frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
