import cv2

IMG_SIZE = 224

def preprocess_image(path):
    img = cv2.imread(path)

    # 1️⃣ Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 2️⃣ Denoise (Gaussian Blur)
    denoised = cv2.GaussianBlur(img, (5,5), 0)

    # 3️⃣ Enhance (CLAHE Contrast Enhancement)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced