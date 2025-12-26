import os
import json
import numpy as np
import cv2
import math
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from deskew import determine_skew


def determine_skew_angle(image):
    grayscale = image
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    if angle is None:
        return 0.0
    return angle

def rotate_image(image, angle, background=(0, 0, 0)):
    old_height, old_width = image.shape[:2]
    angle_radian = math.radians(angle)
    
    width = abs(np.cos(angle_radian) * old_width) + abs(np.sin(angle_radian) * old_height)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = (old_width / 2, old_height / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    rot_mat[0, 2] += (width - old_width) / 2
    rot_mat[1, 2] += (height - old_height) / 2
    
    warped_image = cv2.warpAffine(image, rot_mat, (int(round(width)), int(round(height))), 
                                 borderValue=background)
    return warped_image, rot_mat

def remove_borders(image):
    # Basic border removal based on finding the largest contour (the receipt)
    original = image.copy()
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, (0, 0, 0, 0)
        
    # Filter small stuff
    total_area = image.shape[0] * image.shape[1]
    candidates = [c for c in contours if cv2.contourArea(c) > 0.05 * total_area]
    
    if not candidates:
        # Fallback to largest if nothing meets threshold
        cnt = max(contours, key=cv2.contourArea)
    else:
        cnt = max(candidates, key=cv2.contourArea)
        
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Margin
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2*margin)
    h = min(image.shape[0] - y, h + 2*margin)
    
    crop = original[y:y+h, x:x+w]
    return crop, (x, y, w, h)

def transform_points(x_list, y_list, matrix=None, crop_offset=None):
    # Apply affine matrix or crop translation to lists of points
    new_x = []
    new_y = []
    
    for xs, ys in zip(x_list, y_list):
        if matrix is not None:
            # [[x, y, 1]] @ matrix.T
            # points: N x 2
            pts = np.column_stack((xs, ys, np.ones(len(xs))))
            transformed = pts @ matrix.T
            new_x.append(transformed[:, 0].tolist())
            new_y.append(transformed[:, 1].tolist())
            
        elif crop_offset is not None:
            cx, cy = crop_offset
            new_x.append([x - cx for x in xs])
            new_y.append([y - cy for y in ys])
            
    return new_x, new_y

def preprocess_and_normalize(image, x_coords, y_coords):
    # 1. Deskew
    angle = determine_skew_angle(image)
    image, rot_mat = rotate_image(image, angle)
    x_coords, y_coords = transform_points(x_coords, y_coords, matrix=rot_mat)
    
    # 2. Crop
    image, (cx, cy, cw, ch) = remove_borders(image)
    if cw > 0 and ch > 0:
        x_coords, y_coords = transform_points(x_coords, y_coords, crop_offset=(cx, cy))
    
    return image, x_coords, y_coords

# ==========================================
# CORD-v2 Processing
# ==========================================

def get_yolo_class(category):
    cat = category.lower()
    if 'store' in cat or 'merchant' in cat:
        return 0 # Merchant
    elif 'date' in cat or 'time' in cat:
        return 1 # Date
    elif 'total.total_price' in cat:
        return 2 # Total
    elif 'menu.nm' in cat:
        return 3 # Item Name
    elif 'menu.price' in cat or 'unit_price' in cat:
        return 4 # Item Price
    else:
        return 5 # Other

def convert_to_yolo_format(xs, ys, img_w, img_h):
    # Normalized center x, center y, w, h
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    
    # Clamp
    xmin = max(0, min(img_w, xmin))
    xmax = max(0, min(img_w, xmax))
    ymin = max(0, min(img_h, ymin))
    ymax = max(0, min(img_h, ymax))
    
    box_w = xmax - xmin
    box_h = ymax - ymin
    
    if box_w <= 0 or box_h <= 0:
        return None
        
    cx = xmin + box_w / 2.0
    cy = ymin + box_h / 2.0
    
    return [cx/img_w, cy/img_h, box_w/img_w, box_h/img_h]

def process_split(split_name, output_root, limit=None):
    print(f"Loading {split_name}...")
    ds = load_dataset("naver-clova-ix/cord-v2", split=split_name)
    
    img_out = output_root / "images" / split_name
    lbl_out = output_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    for i, item in tqdm(enumerate(ds)):
        if limit and i >= limit:
            break
            
        try:
            # Load Image
            pil_img = item['image']
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Parse Ground Truth
            gt_text = item['ground_truth']
            gt = json.loads(gt_text)
            valid_lines = gt.get('valid_line', [])
            
            x_raw = []
            y_raw = []
            cats = []
            
            for line in valid_lines:
                cat = line['category']
                for word in line.get('words', []):
                    q = word.get('quad', {})
                    if q:
                        # x1, x2, x3, x4
                        xs = [q['x1'], q['x2'], q['x3'], q['x4']]
                        ys = [q['y1'], q['y2'], q['y3'], q['y4']]
                        x_raw.append(xs)
                        y_raw.append(ys)
                        cats.append(cat)
            
            if not x_raw:
                skipped_count += 1
                continue
                
            # EXECUTE PIPELINE
            final_img, final_x, final_y = preprocess_and_normalize(img, x_raw, y_raw)
            
            # Save
            h, w = final_img.shape[:2]
            if h == 0 or w == 0:
                skipped_count += 1
                continue
                
            cv2.imwrite(str(img_out / f"{i}.jpg"), final_img)
            
            with open(lbl_out / f"{i}.txt", "w") as f:
                for j, (fx, fy) in enumerate(zip(final_x, final_y)):
                    yolo = convert_to_yolo_format(fx, fy, w, h)
                    if yolo:
                        cls_id = get_yolo_class(cats[j])
                        f.write(f"{cls_id} {' '.join(f'{v:.6f}' for v in yolo)}\n")
                        
            processed_count += 1
            
        except Exception as e:
            print(f"Failed {i}: {e}")
            skipped_count += 1
            
    print(f"Split {split_name}: Processed {processed_count}, Skipped {skipped_count}")

if __name__ == "__main__":
    root = Path("datasets/cord_yolo_v2")
    if root.exists():
        # shutil.rmtree(root) 
        pass
        
    # Process
    # Remove 'limit' argument to process full dataset
    process_split("train", root, limit=None)
    process_split("validation", root, limit=None)
    process_split("test", root, limit=None) 
