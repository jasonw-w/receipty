import cv2
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Config
DATASET_ROOT = Path("datasets/cord_yolo_v2")
SPLIT = "train"
NUM_SAMPLES = 6 # Can make a 2x3 grid
OUTPUT_DIR = Path("visualizations/yolo_check")

# Visual config
CLASS_NAMES = {
    0: "Merchant",
    1: "Date",
    2: "Total",
    3: "Item",
    4: "Price",
    5: "Other"
}
# Normalized 0-1 colors for matplotlib
COLORS = {
    0: (1.0, 0.0, 0.0),    # Red
    1: (0.0, 1.0, 0.0),    # Green
    2: (0.0, 0.0, 1.0),    # Blue
    3: (1.0, 1.0, 0.0),    # Yellow
    4: (1.0, 0.0, 1.0),    # Magenta
    5: (0.5, 0.5, 0.5)     # Gray
}

def load_yolo_label(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: 
            continue
            
        cls_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        # De-normalize
        x_center = cx * img_w
        y_center = cy * img_h
        width = w * img_w
        height = h * img_h
        
        x1 = x_center - width/2
        y1 = y_center - height/2
        
        boxes.append((cls_id, x1, y1, width, height))
    return boxes

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    img_dir = DATASET_ROOT / "images" / SPLIT
    lbl_dir = DATASET_ROOT / "labels" / SPLIT
    
    image_files = list(img_dir.glob("*.jpg"))
    if not image_files:
        print(f"No images found in {img_dir}")
        return

    samples = random.sample(image_files, min(len(image_files), NUM_SAMPLES))
    
    print(f"Visualizing {len(samples)} samples using matplotlib...")
    
    # Setup grid
    cols = 3
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(samples):
        ax = axes[idx]
        
        # Load image (OpenCV is BGR, matplotlib needs RGB)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        ax.imshow(img_rgb)
        ax.set_title(img_path.name)
        ax.axis('off')
        
        # Load labels
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        boxes = load_yolo_label(lbl_path, w, h)
        
        for cls_id, x, y, bw, bh in boxes:
            color = COLORS.get(cls_id, (1.0, 1.0, 1.0))
            
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), bw, bh, linewidth=0.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label text
            # label_text = CLASS_NAMES.get(cls_id, str(cls_id))
            # ax.text(x, y - 5, label_text, color=color, fontsize=8, fontweight='bold', 
            #         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
            
    # Hide unused subplots
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()