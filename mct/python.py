# app.py
from flask import Flask, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load face recognition model
face_recognition_model = np.load('face_recognition_model.npy')

@app.route('/verify', methods=['POST'])
def verify_face():
    face_descriptor = request.get_json()['faceDescriptor']
    # Calculate similarity between face descriptor and face recognition model
    similarity = cosine_similarity([face_descriptor], face_recognition_model)
    if similarity > 0.5:
        return jsonify({'verified': True})
    else:
        return jsonify({'verified': False})

if __name__ == '__main__':
    app.run(debug=True)