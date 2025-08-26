import cv2
import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import pickle
import os

# ----------------------------
# Load FaceNet Model with fallback
# ----------------------------
def load_facenet_model(model_path="models/facenet_keras.h5"):
    """
    Load FaceNet model with fallback to dummy model if original fails
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("[INFO] FaceNet model loaded successfully.")
        return model
    except Exception as e:
        print(f"[WARNING] Failed to load FaceNet model: {e}")
        print("[INFO] Creating dummy FaceNet model for testing...")
        
        # Create a simple dummy model that outputs 128-dimensional embeddings
        from tensorflow.keras import layers, models
        dummy_model = models.Sequential([
            layers.Input(shape=(160, 160, 3)),
            layers.Conv2D(32, 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='tanh'),  # FaceNet typically uses tanh for final layer
            layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))  # L2 normalize embeddings
        ])
        
        print("[INFO] Dummy FaceNet model created successfully.")
        return dummy_model

# ----------------------------
# Preprocess face image for FaceNet
# Resize to (160,160), standardize pixel values
# ----------------------------
def preprocess_face(img):
    img = cv2.resize(img, (160,160))
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)
    return img

# ----------------------------
# Get embedding vector from face image
# ----------------------------
def get_face_embedding(model, face_img):
    processed_img = preprocess_face(face_img)
    embedding = model.predict(processed_img, verbose=0)
    return embedding[0]

# ----------------------------
# Load registered embeddings database
# embeddings.pkl: dict{name: embedding}
# ----------------------------
def load_embeddings_db(embeddings_path="data/embeddings/embeddings.pkl"):
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            embeddings_db = pickle.load(f)
        print("[INFO] Embeddings database loaded.")
    else:
        embeddings_db = {}
        print("[INFO] No embeddings database found. Starting fresh.")
    return embeddings_db

# ----------------------------
# Calculate cosine similarity between two embeddings
# ----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (LA.norm(a) * LA.norm(b))

# ----------------------------
# Recognize face by comparing embedding with database
# Returns: best_match_name, best_score
# ----------------------------
def recognize_face(embedding, embeddings_db, threshold=0.5):
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

# ----------------------------
# Add new face to embeddings database
# ----------------------------
def add_face_to_db(name, embedding, embeddings_db, save_path="data/embeddings/embeddings.pkl"):
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
    # Test model loading
    model = load_facenet_model()
    embeddings_db = load_embeddings_db()
    print(f"Model input shape: {model.input_shape}")
    print(f"Embeddings database has {len(embeddings_db)} entries")
