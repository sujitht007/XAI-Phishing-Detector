import os
import sys
import joblib
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

from src.features import extract_features_from_url, FEATURE_COLUMNS

MODELS_PATH = os.path.join(BASE_DIR, 'models', 'saved_models.pkl')
X_TRAIN_PATH = os.path.join(BASE_DIR, 'models', 'X_train.pkl')

models = joblib.load(MODELS_PATH)

urls = [
    'http://konu.edu',
    'http://konu.edu/login',
    'aa',
    'http://192.168.1.1/login',
]

for url in urls:
    feats = extract_features_from_url(url)
    x = pd.DataFrame([feats], columns=FEATURE_COLUMNS)
    print('\nURL:', url)
    print('Features:', feats)
    for name, model in models.items():
        pred = int(model.predict(x)[0])
        prob = float(model.predict_proba(x)[0].max()) if hasattr(model, 'predict_proba') else None
        print(f"  {name}: prediction={'Phishing' if pred==1 else 'Legit'}, prob={prob}")
