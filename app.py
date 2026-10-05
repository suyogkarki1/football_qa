"""Web app for the football QA model.

Serves web/index.html and answers questions at POST /api/ask using the
pretrained sentence-embedding model from embeddings.ipynb.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'football_qa_dataset_full.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')
WEB_DIR = os.path.join(BASE_DIR, 'web')

ENCODER_NAME = 'all-mpnet-base-v2'
C = 100  # best value on the validation set in embeddings.ipynb
THRESHOLD = 0.1  # below this confidence the app answers "I don't know"

encoder = SentenceTransformer(ENCODER_NAME)


def embed(questions):
    return encoder.encode(list(questions), normalize_embeddings=True, batch_size=64)


def load_or_train():
    # the notebook holds out a test set to measure accuracy; the app trains on
    # every question so it knows as many facts as possible
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    print('Training model on all questions (first run only)...')
    data = pd.read_csv(DATA_PATH)
    model = LogisticRegression(C=C, max_iter=3000)
    model.fit(embed(data['question']), data['answer'])
    joblib.dump(model, MODEL_PATH)
    return model


model = load_or_train()
app = Flask(__name__)


def pretty(answer):
    return answer.replace('-', ' ')


@app.get('/')
def index():
    return send_from_directory(WEB_DIR, 'index.html')


@app.post('/api/ask')
def ask():
    question = (request.get_json(silent=True) or {}).get('question', '').strip()
    if not question:
        return jsonify(error='Ask a question first.'), 400

    proba = model.predict_proba(embed([question]))[0]
    top = np.argsort(proba)[::-1][:3]
    confidence = float(proba[top[0]])

    return jsonify(
        answer=pretty(model.classes_[top[0]]) if confidence >= THRESHOLD else None,
        confidence=confidence,
        alternatives=[{'answer': pretty(model.classes_[i]), 'confidence': float(proba[i])} for i in top],
    )


if __name__ == '__main__':
    app.run(debug=False)
