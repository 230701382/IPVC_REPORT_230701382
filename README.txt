=========================================================
GREENHOUSE CROP GROWTH MONITORING PROJECT
=========================================================
Author: Viknesh Kumar M N
Subject: Computer Vision & Image Processing
Project Title: Greenhouse Crop Growth Monitoring
Algorithm: Excess Green Index (ExG) + Otsu Segmentation + ML Classifier
=========================================================


QUICK START
---------------------------------------------------------
Install dependencies

pip install -r requirements.txt

Run the web application

python app.py

Open in browser

http://127.0.0.1:5000

Upload any plant image and get prediction

INSTALLATION

Ensure Python 3.9+ is installed.

Install required libraries:

pip install opencv-python numpy scikit-learn flask matplotlib

Libraries used:

OpenCV → image processing
NumPy → numerical operations
scikit-learn → ML prediction
Flask → web interface
Matplotlib → visualization


PROJECT STRUCTURE
---------------------------------------------------------
Greenhouse-Crop-Monitor/
│
├── app.py               → Flask web application
├── model.py             → ML prediction model
├── preprocessing.py     → Image preprocessing pipeline
├── features.py          → ExG + Otsu feature extraction
├── templates/
│     └── index.html     → Web interface
├── static/              → Saved result images
└── requirements.txt
HOW THE SYSTEM WORKS

The system predicts crop health from greenhouse images using computer vision and machine learning.

Predictions:

• Poor Growth
• Moderate Growth
• Healthy Crop


ALGORITHM EXPLAINED
---------------------------------------------------------
Step 1 — Image Preprocessing

• Resize image to fixed size
• Convert BGR → RGB
• Noise reduction using Gaussian Blur

Purpose: Standardizes input images for consistent analysis.

Step 2 — Excess Green Index (ExG)

ExG highlights vegetation by amplifying green color.

Formula:

ExG = 2G − R − B

Why ExG?
• Plants reflect more green light
• Separates vegetation from soil/background
• Works well in greenhouse environments

Step 3 — Otsu Threshold Segmentation

Automatically separates plant pixels vs background.

Otsu method:
• Finds optimal threshold automatically
• Converts ExG image → binary mask

Output:
White → vegetation 
Black → background 

Step 4 — Vegetation Coverage Calculation

We compute percentage of plant pixels:

Coverage = (Plant Pixels / Total Pixels) × 100

This becomes our main feature for ML prediction.

Step 5 — Machine Learning Prediction

We use Logistic Regression Classifier.

Input:
Vegetation Coverage %

Output:

Coverage %	Prediction
0 – 10%	Poor Growth 
10 – 30%	Moderate Growth 
> 30%	Healthy Crop 

Why ML?
• Provides data-driven prediction
• Easy to extend with more features later
• Demonstrates full AI pipeline


WEB APPLICATION
---------------------------------------------------------
Users can:

Upload plant image
System processes image
Displays:
Segmented vegetation mask
Vegetation coverage %
Growth prediction

This makes the system interactive and user-friendly.

SAMPLE OUTPUT

After uploading image:

Example results:

Prediction: Healthy Crop 
Vegetation Coverage: 42.36%

Result image saved automatically in static/ folder.


PROJECT OBJECTIVES
---------------------------------------------------------
• Monitor greenhouse crop growth automatically
• Reduce manual inspection effort
• Provide early detection of poor vegetation
• Demonstrate Computer Vision + ML integration


FUTURE IMPROVEMENTS
---------------------------------------------------------
• Add deep learning plant disease detection
• Time-series growth monitoring
• Mobile app integration
• Real-time camera feed support
• Multiple crop classification

REPORT TIPS FOR SUBMISSION
---------------------------------------------------------
Include screenshots of:
• Original uploaded image
• ExG segmentation result
• Web app prediction output

Discuss:
• Why ExG works for vegetation detection
• Importance of automated greenhouse monitoring
• Advantages of ML prediction
