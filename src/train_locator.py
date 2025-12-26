from ultralytics import YOLO

def main():
    # 1. Load Nano model
    model = YOLO('yolov8n.pt')
    
    # 2. Train on Synthetic Locator Dataset
    print("Starting Receipt Locator Training...")
    model.train(
        data='datasets/synth_locator/data.yaml',
        epochs=30,             # 30 epochs is plenty for this simple task
        imgsz=640,
        batch=16,
        project='result',
        name='receipt_locator',
        exist_ok=True,
        mosaic=0.5,            # Low mosaic to avoid confusion
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.0,            # Receipts don't flip
        weight_decay=0.001,    # Increased from 0.0005 to reduce overfitting
        lr0=0.001             # Decreased from 0.01 to reduce overfitting
    )
    
    print("Locator training complete.")

if __name__ == "__main__":
    main()
