import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset.csv"
MODEL_FILE = BASE_DIR / "sentiment_model.pkl"

def main():
    data = pd.read_csv(DATA_FILE)
    data = data.dropna(subset=["text", "sentiment"])
    data["text"] = data["text"].astype(str)
    X = data["text"]
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2), max_features=5000, sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    cv = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"5-fold cross-validation accuracy: {cv.mean():.2%} (+/- {cv.std():.2%})")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nMODEL EVALUATION")
    print("----------------")
    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples:  {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: negative, neutral, positive):")
    print(confusion_matrix(y_test, predictions, labels=["negative", "neutral", "positive"]))

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)
    print(f"\nTrained model saved to: {MODEL_FILE.name}")

if __name__ == "__main__":
    main()
