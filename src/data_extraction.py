from src.img_preprocess import *
from src.predict import predict_and_plot
import matplotlib.pyplot as plt
import cv2
import numpy as np
import src.ocr_engine as ocr_engine
import argparse
import json


class DataExtractor:
    def __init__(self, yolo_model=None, ocr_engine=None):
        self.yolo_model = yolo_model
        self.ocr_engine = ocr_engine

    def extract_data_from_img(self, img, Test=False):
        img_processed = preprocess(img)

        # Show Processed
        if Test:
            plt.figure(figsize=(10,10))
            plt.title("Preprocessed Image")
            plt.imshow(img_processed, cmap='gray')
            plt.show()

        from src.ocr_engine import extract_text_from_result

        # Run YOLO Prediction
        print("Running Prediction on Preprocessed Image...")
        # Pass cached YOLO model if available
        annotated_img, result = predict_and_plot(img_processed, model=self.yolo_model, Test=Test)
        if Test:
            plt.imshow(annotated_img[..., ::-1]) # RGB switch for plt if needed (plot() returns BGR)
            plt.show()  
        if result:
            # Fix: predict_and_plot returns a list of results when Test=False
            if isinstance(result, list):
                result = result[0]

            print("Running OCR on detected boxes...")
            # Get class names from model
            names = result.names
            
            # Sort and Extract
            # Pass cached OCR engine if available
            ocr_data = extract_text_from_result(img_processed, result, names, engine=self.ocr_engine)
            
            # Group by Lines for better context
            from src.ocr_engine import group_by_lines
            lines = group_by_lines(ocr_data)
            
            # Visualize Individual Crops (Top 20) for debug
            if Test:    
                if len(ocr_data) > 0:
                    num_items = min(len(ocr_data), 20)
                    cols = 5
                    rows = math.ceil(num_items / cols)
                
                plt.figure(figsize=(15, 3 * rows))
                plt.suptitle("Individual OCR Results (Top 20)")
                
                for i in range(num_items):
                    item = ocr_data[i]
                    x1, y1, x2, y2 = item['box']
                    crop = img_processed[y1:y2, x1:x2]
                    disp_crop = self.ocr_engine.preprocess_for_ocr(crop)
                    plt.subplot(rows, cols, i+1)
                    plt.imshow(disp_crop, cmap='gray')
                    plt.title(f"{item['class_name']}\n'{item['text']}'", fontsize=3)
                    plt.axis('off')
                
                plt.tight_layout()
                plt.show()
            
            # Write Result grouped by lines
            with open("result.txt", "w", encoding="utf-8") as f:
                full_receipt_text = ""
                for line_items in lines:
                    # Join items in the same line with " " (double space)
                    line_str = " ".join([f"{item['text']}" for item in line_items])
                    f.write(line_str + "\n")
                    full_receipt_text += line_str + "\n"
                    print(line_str) # Print structured line to console
                        
            # --- LLM Parsing ---
            from src.LLM import LLM
            import ast
            import csv

            print("\n--- Sending to LLM for Parsing ---")
            try:
                response_text = LLM(full_receipt_text)
                print("LLM Response:")
                # Parse JSON
                try:
                    # Clean potential markdown
                    clean_text = response_text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean_text)
                    
                    # Save JSON
                    with open("result/parsed_receipt.json", "w") as f:
                        json.dump(data, f, indent=4)
                    print(data)
                    print(f"Saved parsed data to result/parsed_receipt.json")
                    
                except json.JSONDecodeError as je:
                    print(f"Error parsing JSON: {je}")
                    print(f"Raw text: {response_text}")
                    data = None

            except Exception as e:
                print(f"LLM Error: {e} (Check API Key)")
                data = None
        else:
             data = None
        return data

if __name__ == "__main__":
    import time
    start_time = time.time()
    extractor = DataExtractor()
    img = cv2.imread("demo/7.jpg")
    extractor.extract_data_from_img(img, Test=False)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.3f} seconds")