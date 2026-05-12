import cv2
import joblib
from preprocessing import preprocess_image
from features import extract_exg_features

model = joblib.load("plant_model.pkl")

img = preprocess_image("test.jpg")
coverage, mask = extract_exg_features(img)

prediction = model.predict([[coverage]])

if prediction == 0:
    label = "Plant Detected 🌿"
else:
    label = "Low Vegetation"

print("Vegetation Coverage:", coverage*100, "%")
print("Prediction:", label)

cv2.imshow("Segmentation Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()