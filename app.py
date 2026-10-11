from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Files downloaded from Colab (keep them next to app.py)
scaler = joblib.load('scaler.pkl')
clf = joblib.load('classifier.pkl')

# Exact column order the scaler was trained on
FEATURES = list(scaler.feature_names_in_)

# Must be IDENTICAL to transform() in the Colab notebook
LOG_COLS = ['income', 'gdpp', 'exports', 'imports', 'health']


def transform(X):
    X = X.copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    X['inflation'] = np.sign(X['inflation']) * np.log1p(np.abs(X['inflation']))
    return X


def to_frame(rows):
    """Turn a list of dicts into a clean DataFrame in training column order."""
    return pd.DataFrame([{f: float(r[f]) for f in FEATURES} for r in rows], columns=FEATURES)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        X = to_frame([data])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Please provide valid numbers for all 9 indicators.'}), 400

    x_scaled = scaler.transform(transform(X))
    pred = clf.predict(x_scaled)[0]
    proba = clf.predict_proba(x_scaled)[0]

    # How much each feature pushed toward the predicted class for THIS country
    idx = list(clf.classes_).index(pred)
    contrib = clf.coef_[idx] * x_scaled[0]

    return jsonify({
        'label': str(pred),
        'confidence': round(float(proba.max()) * 100, 1),
        'probabilities': {str(c): round(float(p) * 100, 1) for c, p in zip(clf.classes_, proba)},
        'contributions': {f: round(float(v), 3) for f, v in zip(FEATURES, contrib)},
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Classify many rows at once. Used by the decision map and the 'flip finder'."""
    data = request.get_json(silent=True) or {}
    rows = data.get('rows', [])
    if not rows or len(rows) > 2000:
        return jsonify({'error': 'Send between 1 and 2000 rows.'}), 400
    try:
        X = to_frame(rows)
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Every row needs all 9 indicators as numbers.'}), 400

    x_scaled = scaler.transform(transform(X))
    labels = clf.predict(x_scaled)
    conf = clf.predict_proba(x_scaled).max(axis=1)

    return jsonify({
        'labels': [str(l) for l in labels],
        'confidence': [round(float(c), 3) for c in conf],
    })


if __name__ == '__main__':
    app.run(debug=True)