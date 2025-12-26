import cv2
import numpy as np
import random
import os
from pathlib import Path
from tqdm import tqdm
import math

def generate_noise_background(width, height):
    """Generate a random noise background."""
    bg = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    # Blur it to look less like static
    bg = cv2.GaussianBlur(bg, (51, 51), 0)
    return bg

def generate_solid_background(width, height):
    """Generate a random solid/gradientish background."""
    color = np.random.randint(0, 256, 3, dtype=np.uint8)
    bg = np.full((height, width, 3), color, dtype=np.uint8)
    return bg

def create_synthetic_sample(receipt_img, bg_size=(640, 640)):
    """
    Paste receipt onto a background with random scale, rotation, and position.
    Returns (composite_img, bbox_yolo).
    """
    bg_w, bg_h = bg_size
    
    # 1. Create Background
    if random.random() > 0.5:
        bg = generate_noise_background(bg_w, bg_h)
    else:
        bg = generate_solid_background(bg_w, bg_h)
        
    h, w = receipt_img.shape[:2]
    
    # 2. Random Scale (0.4 to 0.8 of background)
    scale = random.uniform(0.4, 0.8)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Keep aspect ratio but fit within background
    if new_w > bg_w: 
        scale = bg_w / w * 0.8
        new_w = int(w * scale)
        new_h = int(h * scale)
    if new_h > bg_h:
        scale = bg_h / h * 0.8
        new_w = int(w * scale)
        new_h = int(h * scale)
        
    resized = cv2.resize(receipt_img, (new_w, new_h))
    
    # 3. Random Rotation (-20 to 20 degrees)
    angle = random.uniform(-20, 20)
    
    # Pad resized image to allow rotation without cutting corners
    diagonal = int(math.sqrt(new_w**2 + new_h**2))
    
    # CRITIAL FIX: Ensure diagonal fits in background
    # If diagonal is too big, rescale again
    max_diag = min(bg_w, bg_h) * 0.9
    if diagonal > max_diag:
        scale_fix = max_diag / diagonal
        new_w = int(new_w * scale_fix)
        new_h = int(new_h * scale_fix)
        resized = cv2.resize(receipt_img, (new_w, new_h))
        diagonal = int(math.sqrt(new_w**2 + new_h**2))
        
    pad_h = (diagonal - new_h) // 2
    pad_w = (diagonal - new_w) // 2
    pad_h = (diagonal - new_h) // 2
    pad_w = (diagonal - new_w) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0,0))
    
    center = (padded.shape[1]//2, padded.shape[0]//2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(padded, rot_mat, (padded.shape[1], padded.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    
    # 4. Find bounding box of non-black pixels (the receipt)
    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_rot, 1, 255, cv2.THRESH_BINARY)
    x, y, w_rot, h_rot = cv2.boundingRect(thresh)
    
    # Extract the actual rotated content
    receipt_final = rotated[y:y+h_rot, x:x+w_rot]
    
    # 5. Paste onto Background at random position
    paste_x = random.randint(0, bg_w - w_rot)
    paste_y = random.randint(0, bg_h - h_rot)
    
    # Create mask for pasting (simple non-black check for rough mask)
    # Ideally should use alpha channel but simplistic approach works for YOLO
    roi = bg[paste_y:paste_y+h_rot, paste_x:paste_x+w_rot]
    
    receipt_gray = cv2.cvtColor(receipt_final, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(receipt_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    bg_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    receipt_fg = cv2.bitwise_and(receipt_final, receipt_final, mask=mask)
    
    dst = cv2.add(bg_bg, receipt_fg)
    bg[paste_y:paste_y+h_rot, paste_x:paste_x+w_rot] = dst
    
    # 6. Calculate YOLO Label (class 0)
    # cx, cy, w, h (normalized)
    cx = (paste_x + w_rot / 2) / bg_w
    cy = (paste_y + h_rot / 2) / bg_h
    nw = w_rot / bg_w
    nh = h_rot / bg_h
    
    return bg, [cx, cy, nw, nh]

def main():
    # Source Images (Existing CORD train set)
    source_dir = Path("datasets/cord_yolo_v2/images/train")
    if not source_dir.exists():
        print(f"Source not found: {source_dir}")
        return
        
    output_root = Path("datasets/synth_locator")
    img_dir = output_root / "images/train"
    lbl_dir = output_root / "labels/train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create val set
    img_val = output_root / "images/val"
    lbl_val = output_root / "labels/val"
    img_val.mkdir(parents=True, exist_ok=True)
    lbl_val.mkdir(parents=True, exist_ok=True)
    
    images = list(source_dir.glob("*.jpg"))
    # Limit for speed (e.g. 500 samples is enough for simple object detection)
    images = images[:500] 
    
    print(f"Generating synthetic dataset from {len(images)} source images...")
    
    for i, img_path in enumerate(tqdm(images)):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Determine split (80/20)
        is_val = random.random() < 0.2
        target_img_dir = img_val if is_val else img_dir
        target_lbl_dir = lbl_val if is_val else lbl_dir
        
        # Generate Sample
        synth_img, label = create_synthetic_sample(img)
        
        # Save Image
        out_name = f"synth_{i}"
        cv2.imwrite(str(target_img_dir / f"{out_name}.jpg"), synth_img)
        
        # Save Label (class 0)
        with open(target_lbl_dir / f"{out_name}.txt", "w") as f:
            f.write(f"0 {label[0]:.6f} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f}\n")
            
    # Create data.yaml
    yaml_content = f"""
path: {output_root.absolute()}
train: images/train
val: images/val
names:
  0: receipt
"""
    with open(output_root / "data.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"Dataset generated at {output_root}")

if __name__ == "__main__":
    main()
