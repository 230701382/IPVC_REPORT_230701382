import os
import cv2
import time
from flask import Flask, render_template, request
from preprocessing import preprocess_image
from features import extract_exg_features
from model import predict_growth

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():
    result = None
    coverage = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        path = "temp.jpg"
        file.save(path)

        # Run pipeline
        img = preprocess_image(path)
        coverage_value, mask = extract_exg_features(img)

        coverage = round(coverage_value * 100, 2)

        # ⭐ Save result with timestamp (prevents caching)
        # Get correct static folder path
        static_folder = os.path.join(os.path.dirname(__file__), "static")

        filename = f"result_{int(time.time())}.png"
        full_path = os.path.join(static_folder, filename)

        # Save image to REAL static folder
        cv2.imwrite(full_path, mask)

        # Path used by HTML
        image_path = f"static/{filename}"

        result = predict_growth(coverage)

    return render_template("index.html",
                           result=result,
                           coverage=coverage,
                           image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)