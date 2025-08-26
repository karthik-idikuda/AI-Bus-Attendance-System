import cv2
import numpy as np
import pickle
import os
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

def recognize_face(embedding, embeddings_db, threshold=0.5):
    """
    Recognize face by comparing embedding with database
    """
    if len(embeddings_db) == 0:
        return "Unknown", 0.0

    best_match_name = "Unknown"
    best_score = 0.0

    for name, stored_embedding in embeddings_db.items():
        # Calculate similarity
        similarity = cosine_similarity(embedding, stored_embedding)
        
        if similarity > best_score:
            best_score = similarity
            best_match_name = name

    # Apply threshold
    if best_score < threshold:
        best_match_name = "Unknown"

    return best_match_name, best_score

def add_face_to_db(name, embedding, embeddings_db, save_path="data/embeddings/embeddings.pkl"):
    """Add new face to embeddings database"""
    embeddings_db[name] = embedding
    
    # Save updated database
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings_db, f)
    
    print(f"[INFO] Added {name} to embeddings database.")

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
