import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data (coverage % vs class)
# 0 = Poor growth
# 1 = Medium growth
# 2 = Healthy growth

X = np.array([
    [2],[5],[8],      # poor
    [12],[18],[25],   # medium
    [35],[45],[60]    # healthy
])

y = np.array([
    0,0,0,
    1,1,1,
    2,2,2
])

model = LogisticRegression()
model.fit(X,y)

def predict_growth(coverage_percent):
    pred = model.predict([[coverage_percent]])[0]

    if pred == 0:
        return "Poor Growth 🟫"
    elif pred == 1:
        return "Moderate Growth 🌿"
    else:
        return "Healthy Crop 🌱"