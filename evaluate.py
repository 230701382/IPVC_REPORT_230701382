import joblib
import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocessing import preprocess_image
from features import extract_exg_features

DATASET_PATH = "/Users/vikneshkumar/Downloads/ipcv/Agricultural-crops"
classes = ["plants", "background"]

model = joblib.load("plant_model.pkl")

X_test = []
y_test = []

print("🔎 Evaluating model...")

for label, cls in enumerate(classes):
    folder = os.path.join(DATASET_PATH, cls)

    for root, dirs, files in os.walk(folder):
        for file in files:
            if not file.lower().endswith(('.png','.jpg','.jpeg')):
                continue

            path = os.path.join(root, file)

            img = preprocess_image(path)
            if img is None:
                continue

            coverage, _ = extract_exg_features(img)

            X_test.append([coverage])
            y_test.append(label)

pred = model.predict(X_test)

print("\n📊 RESULTS")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))