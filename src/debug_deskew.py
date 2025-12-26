import cv2
import numpy as np
import os
from img_preprocess import smart_deskew_and_crop, four_point_transform

def debug_preprocessing(image_path):
    print(f"Debugging {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    original = image.copy()
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    cv2.imwrite("result/debug_edges.jpg", edged)
    print("Saved result/debug_edges.jpg")
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    debug_cnt = original.copy()
    cv2.drawContours(debug_cnt, contours, -1, (0, 255, 0), 2)
    
    receipt_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            receipt_contour = approx
            cv2.drawContours(debug_cnt, [receipt_contour], -1, (0, 0, 255), 4) # Red for found
            print("Found 4-point contour")
            break
            
    if receipt_contour is None and len(contours) > 0:
        print("Fallback to minAreaRect")
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        receipt_contour = np.int0(box)
        cv2.drawContours(debug_cnt, [receipt_contour], -1, (255, 0, 0), 4) # Blue for fallback

    cv2.imwrite("result/debug_contours.jpg", debug_cnt)
    print("Saved result/debug_contours.jpg")

    if receipt_contour is not None:
        pts = receipt_contour.reshape(4, 2)
        warped = four_point_transform(original, pts)
        cv2.imwrite("result/debug_warped.jpg", warped)
        print(f"Saved result/debug_warped.jpg (shape: {warped.shape})")
    else:
        print("No receipt contour found at all.")

if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    debug_preprocessing("demo/2.jpg")
