import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import preprocess_image
from features import extract_exg_features

# 📁 Your dataset path (Mac)
DATASET_PATH = "/Users/vikneshkumar/Downloads/ipcv/Agricultural-crops"

classes = ["plants", "background"]

X = []
y = []

print("📸 Loading dataset...")

for label, cls in enumerate(classes):
    folder = os.path.join(DATASET_PATH, cls)

    # walk through ALL subfolders recursively 🔥
    for root, dirs, files in os.walk(folder):

        for file in files:
            if not file.lower().endswith(('.png','.jpg','.jpeg')):
                continue

            path = os.path.join(root, file)

            img = preprocess_image(path)
            if img is None:
                print("Skipped:", path)
                continue

            coverage, mask = extract_exg_features(img)

            X.append([coverage])
            y.append(label)
X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# 🤖 Train KNN Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

print("✅ Model trained successfully!")

# 💾 Save model
joblib.dump(model, "plant_model.pkl")
print("💾 Model saved as plant_model.pkl")