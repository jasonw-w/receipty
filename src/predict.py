from ultralytics import YOLO
import glob
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random
from pathlib import Path
import numpy as np

# Config
MODEL_PATH = "result/receipt_yolov8n/weights/best.pt"
CONF_THRESH = 0.25

# CORD Classes
CLASS_NAMES = {
    0: "Merchant",
    1: "Date",
    2: "Total",
    3: "Item",
    4: "Price",
    5: "Other"
}
# Consistent Colors (0-1 RGB)
COLORS = {
    0: (1.0, 0.0, 0.0),   # Red
    1: (0.0, 1.0, 0.0),   # Green
    2: (0.0, 0.0, 1.0),   # Blue
    3: (1.0, 1.0, 0.0),   # Yellow
    4: (1.0, 0.0, 1.0),   # Magenta
    5: (0.5, 0.5, 0.5)    # Gray
}

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return None
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def predict_and_plot(source, model=None, Test=False):
    """
    Run prediction on a source (path or numpy array) and plot result.
    """
    if model is None:
        model = load_model()
        if model is None: return

    # Pre-process source if it's a numpy array to ensure 3 channels
    if isinstance(source, np.ndarray):
        # If grayscale (2D), convert to BGR
        if len(source.shape) == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        elif len(source.shape) == 3 and source.shape[2] == 1:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

    # Run inference
    # If source is a single numpy array (image), list wrap it or pass directly
    results = model.predict(source=source, save=False, conf=CONF_THRESH)
    
    # Plotting logic
    # If single image, just plot it
    if Test:
        if len(results) == 1:
            result = results[0]
            
            # Setup Figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Image
        if isinstance(source, (str, Path)):
             # It loaded the image from disk, result.orig_img is BGR
             img_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        elif isinstance(source, np.ndarray):
             # Passed an array. If grayscale (2D), convert to RGB for plotting
             if len(source.shape) == 2:
                 img_rgb = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
             else:
                 # Assume BGR if passing to YOLO usually, but let's check
                 # YOLO usually works on BGR arrays
                 img_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)

        ax.imshow(img_rgb)
        ax.axis('off')

        # Boxes
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2-x1, y2-y1
            
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            label_text = CLASS_NAMES.get(cls_id, str(cls_id))
            color = COLORS.get(cls_id, (1, 1, 1))
            
            # Rect
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Label
            ax.text(x1, y1-5, f"{label_text} {conf:.2f}", color='white', fontsize=3, fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.3, edgecolor='none', pad=1))
        
        plt.show()
        
        # Return the annotated image (numpy array) and result object
        return result.plot(), result
    else:
        # Multiple images (batch) - return list of results
        return [r.plot() for r in results], results

def main():
    # Test on demo folder
    test_dir = Path("demo")
    if not test_dir.exists(): return
    images = list(test_dir.glob("*.jpg"))
    if images:
        # Predict on first 4
        predict_and_plot([str(p) for p in images[:4]])

if __name__ == "__main__":
    main()
