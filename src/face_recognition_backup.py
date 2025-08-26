import cv2
import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import pickle
import os

# ----------------------------
# Load FaceNet Model
# ----------------------------
def load_facenet_model(model_path="models/facenet_keras.h5"):
    model = tf.keras.models.load_model(model_path)
    print("[INFO] FaceNet model loaded successfully.")
    return model

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
    preprocessed = preprocess_face(face_img)
    embedding = model.predict(preprocessed)
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
    best_match_name = "Unknown"
    best_score = -1

    for name, db_emb in embeddings_db.items():
        score = cosine_similarity(embedding, db_emb)
        if score > best_score:
            best_score = score
            best_match_name = name

    if best_score >= threshold:
        return best_match_name, best_score
    else:
        return "Unknown", best_score

# ----------------------------
# Example usage (testing)
# ----------------------------
if __name__ == "__main__":
    # Load FaceNet model
    facenet_model = load_facenet_model()

    # Load embeddings database
    embeddings_db = load_embeddings_db()

    # Load test image
    test_img_path = "data/faces/student1/img1.jpg"
    img = cv2.imread(test_img_path)

    # Get embedding
    embedding = get_face_embedding(facenet_model, img)

    # Recognize
    name, score = recognize_face(embedding, embeddings_db)
    print(f"Recognition Result: {name} (Score: {score:.3f})")
