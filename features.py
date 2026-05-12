import cv2
import numpy as np

def extract_exg_features(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = img_rgb[:,:,0].astype(float)
    G = img_rgb[:,:,1].astype(float)
    B = img_rgb[:,:,2].astype(float)

    # 🌿 Excess Green Index
    exg = 2*G - R - B
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    exg = exg.astype(np.uint8)

    # 🌿 Otsu Segmentation
    _, mask = cv2.threshold(exg, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 🌿 Feature = vegetation coverage %
    plant_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    coverage = plant_pixels / total_pixels

    return coverage, mask