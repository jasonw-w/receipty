import os
import cv2
import numpy as np
import math
from deskew import determine_skew
def inverse_image(image):
    """Invert image colors"""
    inverted = cv2.bitwise_not(image)
    return inverted

def gray_scale(image):
    """Convert image to grayscale"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        print("Image is already grayscale")
    return gray

def noise_removal(image, kernel_size=(2, 2), iterations=2):
    """Remove noise using morphological operations"""
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    eroded = cv2.erode(dilated, kernel, iterations=iterations)
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    denoised = cv2.medianBlur(closed, 3)
    return denoised

def deskew(image, rotation_matrix=None, return_angle=False):
    """Deskew image using automatic skew detection"""
    if rotation_matrix is None:
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    background = (255, 255, 255)
    angle = determine_skew(image)
    old_height, old_width = image.shape[:2]
    angle_radian = math.radians(angle)
    
    # Calculate new dimensions after rotation
    width = abs(np.cos(angle_radian) * old_width) + abs(np.sin(angle_radian) * old_height)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    # Get rotation matrix from OpenCV
    image_center = (old_width / 2, old_height / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    # Adjust translation to center the rotated image in new canvas
    rot_mat[0, 2] += (width - old_width) / 2
    rot_mat[1, 2] += (height - old_height) / 2
    
    # Apply transformation
    warped_image = cv2.warpAffine(image, rot_mat, (int(round(width)), int(round(height))), 
                                 borderValue=background)
    
    if return_angle:
        return warped_image, rot_mat, angle
    else:
        return warped_image, rot_mat

def remove_borders(image, total_removed=(0, 0, 0, 0)):
    """Remove borders using contour detection"""
    original = image.copy()
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = adaptive_threshold(binary, blockSize=11, C=2)
    
    # Find contours on the binary image
    try:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        # OpenCV 3 compatibility
        _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Handle empty contours case
    if not contours:
        print("No contours found")
        return image, (0, 0, 0, 0)
    
    # Filter out small contours (noise)
    min_area = image.shape[0] * image.shape[1] * 0.02
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not contours:
        print("No significant contours found after filtering")
        return image, (0, 0, 0, 0)
    
    # Get largest contour
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Add small margin if possible
    margin_x = int(w * 0.01)
    margin_y = int(h * 0.01)
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(image.shape[1] - x, w + 2*margin_x)
    h = min(image.shape[0] - y, h + 2*margin_y)
    
    # Crop from original image
    crop = original[y:y+h, x:x+w]
    return crop, (total_removed[0] + x, total_removed[1] + y, w, h)

def threshold(image, thresh=200, maxval=255, type=cv2.THRESH_BINARY):
    """Apply basic threshold"""
    thresh_val, thresholded = cv2.threshold(image, thresh, maxval, type)
    return thresholded

def histogram_threshold_otsu(image):
    """Apply Otsu's automatic threshold selection"""
    # Ensure image is grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh_value, thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_img, thresh_value

def histogram_threshold_triangle(image):
    """Apply Triangle automatic threshold selection"""
    # Ensure image is grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh_value, thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return thresh_img, thresh_value

def compare_threshold_data_loss(image, threshold_values=[50, 100, 127, 150, 200]):
    """Compare how different threshold values affect data retention"""
    # Ensure grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    results = []
    total_pixels = image.size
    
    for thresh_val in threshold_values:
        # Apply threshold
        _, thresh_img = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Calculate statistics
        white_pixels = np.sum(thresh_img == 255)
        black_pixels = np.sum(thresh_img == 0)
        
        data_retained_percent = (white_pixels / total_pixels) * 100
        data_lost_percent = (black_pixels / total_pixels) * 100
        
        results.append({
            'threshold': thresh_val,
            'image': thresh_img,
            'data_retained_percent': data_retained_percent,
            'data_lost_percent': data_lost_percent,
            'white_pixels': white_pixels,
            'black_pixels': black_pixels
        })
    
    return results

def adaptive_threshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                      thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2):
    """Apply adaptive threshold"""
    adaptive = cv2.adaptiveThreshold(image, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    return adaptive

def CLAHE(image, clipLimit=2.0, tileGridSize=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    # Ensure image is grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    enhanced = clahe.apply(image)
    _, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def dilate(image, iteration=1):
    """Dilate image"""
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(image, kernel, iteration)
    return dilated

def erode(image, iteration=2):
    """Erode image"""
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iteration)
    return eroded

def fix_image(image, original_image):
    """Fix image using morphological operations"""
    # Ensure original_image is grayscale
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply OTSU threshold
    thresh = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Invert the image
    removed = 255 - image

    # Ensure removed is grayscale before dilation
    if len(removed.shape) == 3 and removed.shape[2] == 3:
        removed = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)

    # Dilate the image
    dilate_img = cv2.dilate(removed, repair_kernel, iterations=5)

    # Bitwise AND with thresholded image
    pre_result = cv2.bitwise_and(dilate_img, thresh)

    # Morphological closing
    result = cv2.morphologyEx(pre_result, cv2.MORPH_CLOSE, repair_kernel, iterations=5)

    # Final bitwise AND with thresholded image
    final = cv2.bitwise_and(result, thresh)
    return final

def save_img(image, path):
    """Save image to specified path"""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array.")
    cv2.imwrite(path, image)
    print(f"Image saved to {path}")

# Complete preprocessing pipeline function
def preprocess_receipt_image(image_path):
    """Complete preprocessing pipeline for receipt images"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    log_process("Starting preprocessing pipeline")
    
    # Step 1: Convert to grayscale
    gray = gray_scale(image)
    
    # Step 2: Apply adaptive threshold
    thresh = adaptive_threshold(gray, blockSize=11, C=7)
    
    # Step 3: First deskew
    deskewed1, rot_mat1, angle1 = deskew(thresh, return_angle=True)
    log_process(f"First deskew: {angle1:.2f} degrees")
    
    # Step 4: Remove borders
    cropped1, border1 = remove_borders(deskewed1)
    
    # Step 5: Second deskew
    deskewed2, rot_mat2, angle2 = deskew(cropped1, return_angle=True)
    log_process(f"Second deskew: {angle2:.2f} degrees")
    
    # Step 6: Final border removal
    final_image, border2 = remove_borders(deskewed2)
    
    log_process("Preprocessing pipeline completed")
    
    return {
        'final_image': final_image,
        'transformations': {
            'rot_mat1': rot_mat1,
            'border1': border1,
            'rot_mat2': rot_mat2,
            'border2': border2,
            'angles': [angle1, angle2]
        }
    }

# Coordinate transformation functions
def apply_rotation_to_coordinates(x_coords, y_coords, rotation_matrix):
    """Apply rotation matrix to coordinate lists"""
    new_x, new_y = [], []
    full_matrix = np.vstack([rotation_matrix, [0, 0, 1]])
    
    for x_coord, y_coord in zip(x_coords, y_coords):
        points = np.vstack([x_coord, y_coord, np.ones(len(x_coord))])
        transformed = full_matrix @ points
        new_x.append(transformed[0, :].tolist())
        new_y.append(transformed[1, :].tolist())
    
    return new_x, new_y

def apply_crop_to_coordinates(x_coords, y_coords, crop_x, crop_y):
    """Apply crop offset to coordinate lists"""
    adjusted_x = [[x - crop_x for x in x_coord] for x_coord in x_coords]
    adjusted_y = [[y - crop_y for y in y_coord] for y_coord in y_coords]
    return adjusted_x, adjusted_y

def clean_specks(img):
    kernel = np.ones((1, 1), np.uint8) # Very small kernel
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def remove_shadows(img):
    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm_img)
    return cv2.merge(result_planes)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    PATH = r"demo\0.jpg"
    img = cv2.imread(PATH)
    img = remove_shadows(img)
    img = gray_scale(img)
    img = inverse_image(img)
    img = deskew(img)[0]
    img = remove_borders(img)[0]
    img = adaptive_threshold(img)
    img = clean_specks(img)
    img = remove_borders(img)[0]
    plt.imshow(img, cmap='gray')
    plt.show()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def smart_deskew_and_crop(image):
    original = image.copy()
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 1. Blur to remove noise (texture)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 2. Thresholding (Otsu) - Detect white receipt against dark background
    # Note: If background is lighter than receipt, this might need inversion. 
    # But usually receipts are white.
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # 3. Morphological Operations to close gaps (text within receipt)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 4. Find Contours
    # Use CLOSED image
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small stuff
    h, w = image.shape[:2]
    min_area = (h * w) * 0.05 # At least 5% of image
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Sort largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    
    receipt_contour = None
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        # 0.02 is standard approximation accuracy
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            receipt_contour = approx
            break
            
    # Fallback to MinAreaRect if exact 4 corners (polygon) not found
    if receipt_contour is None and len(contours) > 0:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        receipt_contour = np.int32(box)

    if receipt_contour is not None:
        pts = receipt_contour.reshape(4, 2)
        warped = four_point_transform(original, pts)
        return warped
    else:
        # If absolutely nothing found, return original
        return image

from ultralytics import YOLO

# Global model cache
LOCATOR_MODEL = None

def get_locator_model():
    global LOCATOR_MODEL
    if LOCATOR_MODEL is None:
        path = "result/receipt_locator/weights/best.pt"
        if os.path.exists(path):
            try:
                LOCATOR_MODEL = YOLO(path)
            except:
                print("Failed to load locator model")
    return LOCATOR_MODEL

def locate_receipt(image):
    """use YOLO to find the receipt bounding box and crop it"""
    model = get_locator_model()
    if model is None:
        return image
        
    # Predict
    results = model.predict(image, conf=0.25, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return image
        
    # Get largest box
    boxes = results[0].boxes
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    max_idx = np.argmax(areas.cpu().numpy())
    
    x1, y1, x2, y2 = boxes.xyxy[max_idx].cpu().numpy().astype(int)
    
    # Pad slightly
    h, w = image.shape[:2]
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]

def preprocess(img):
    # 1. Coarse Localization (YOLO)
    # Finds the general area of the receipt
    img = locate_receipt(img)

    # 2. Fine Deskew (Perspective Transform)
    # Finds exact corners and straightens
    img = smart_deskew_and_crop(img)
    
    # 3. Shadow Removal
    img = remove_shadows(img)
    
    return img