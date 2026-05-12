import cv2
import joblib
import sys
from preprocessing import preprocess_image
from features import extract_exg_features

# Load trained model
model = joblib.load("plant_model.pkl")

def predict_image(image_path):
    # Preprocess
    img = preprocess_image(image_path)
    if img is None:
        print("❌ Could not load image")
        return

    # Feature extraction
    coverage, mask = extract_exg_features(img)

    # Prediction
    pred = model.predict([[coverage]])[0]

    if pred == 0:
        label = "🌿 Plant Detected"
    else:
        label = "🟫 Low Vegetation"

    print("\n📷 Image:", image_path)
    print("Vegetation Coverage:", round(coverage*100,2), "%")
    print("Prediction:", label)

    # Show segmentation result
    cv2.imshow("Segmentation Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 📌 take image from terminal argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
    else:
        predict_image(sys.argv[1])