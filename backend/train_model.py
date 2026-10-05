import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from knowledge import PROFILES

def build_dataset(n=700, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for disease, probs in PROFILES.items():
        p = np.array(probs)
        rows = (rng.random((n, len(p))) < p).astype(int)
        X.append(rows); y += [disease] * n
    return np.vstack(X), np.array(y)

def train(path="model.pkl"):
    X, y = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=1).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(yte, pred))
    joblib.dump(clf, path)
    return acc

if __name__ == "__main__":
    train()
