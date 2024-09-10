# train_model.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

# Load face descriptors from database
face_descriptors = []
with open('database.sql', 'r') as f:
    for line in f:
        face_descriptor = line.strip().split(',')[1]
        face_descriptors.append(np.fromstring(face_descriptor, dtype=np.float32))

# Normalize face descriptors
normalizer = Normalizer()
face_descriptors = normalizer.fit_transform(face_descriptors)

# Train face recognition model
face_recognition_model = np.mean(face_descriptors, axis=0)

# Save face recognition model
np.save('face_recognition_model.npy', face_recognition_model)