from ultralytics import YOLO
import os

def main():
    # 1. Load the model
    # We use 'yolov8n.pt' which is the Nano model (smallest & fastest).
    # This automatically downloads the pre-trained weights if not present.
    print("Loading YOLOv12n model...")
    model = YOLO('yolo12n.pt') 

    # 2. Train the model
    # We point to 'v2/data.yaml' for the dataset config.
    # We disable 'mosaic' augmentation (mosaic=0.0) beacuse it's bad for document layout.
    print("Starting training on CORD-v2 dataset...")
    
    # Check if data.yaml exists
    if not os.path.exists("v2/data.yaml"):
        # Fallback to absolute path or just data.yaml if running from v2
        if os.path.exists("data.yaml"):
            data_path = "data.yaml"
        else:
            raise FileNotFoundError("Could not find v2/data.yaml")
    else:
        data_path = "v2/data.yaml"

    results = model.train(
        data=data_path,
        epochs=75,           # Good starting point for 800 images
        imgsz=640,           # Standard resolution
        batch=16,            # Adjust if you run out of GPU memory (use 8 or 4)
        name='receipt_yolov12n', # Name of the experiment folder in 'runs/'
        lr0=0.001,
        dropout=0.01,
        cls=3,
        box=7.5,
        # Hyperparameters for Documents
        mosaic=0.0,          # DISABLE mosaic (mixing 4 images). Critical for receipts.
        degrees=12,         # Very slight rotation (we already deskewed)
        translate=0.1,       # Slight translation
        scale=0,           # Scale variation (zoom in/out)
        fliplr=0.0,          # No horizontal flipping (text becomes backwards)
        
        project='result',    # Save outputs to 'result/' instead of 'runs/'
        exist_ok=True,       # Overwrite if exists (useful for restarting)
        plots=True,
        save_period=1,
        device=0
    )
    
    print("Training finished!")
    print(f"Best weights saved at: {results.save_dir}/weights/best.pt")
    
    # 3. Export (optional)
    # model.export(format='onnx')

if __name__ == "__main__":
    main()
