import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:
    has_xgb = False


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'data', 'phishing_email_dataset_20000.csv')
OUT_DIR = os.path.join(BASE, 'models')
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    # Expect last column 'label' with 1=phishing, 0=legit
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y


def train_and_save():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['LogisticRegression'] = lr

    # XGBoost if available
    if has_xgb:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb

    # Save models dict and X_train for explainability
    models_path = os.path.join(OUT_DIR, 'email_saved_models.pkl')
    x_train_path = os.path.join(OUT_DIR, 'email_X_train.pkl')
    joblib.dump(models, models_path)
    joblib.dump(X_train, x_train_path)

    print(f"Saved {len(models)} models to {models_path}")


if __name__ == '__main__':
    train_and_save()
