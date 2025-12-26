import cv2

import numpy as np
try:
    # Recommended for production/cloud (uses rapidocr-onnxruntime)
    from rapidocr_onnxruntime import RapidOCR
except ImportError as e:
    print(f"DEBUG: Could not import rapidocr_onnxruntime: {e}")
    # Fallback for local dev (uses rapidocr)
    from rapidocr import RapidOCR

# Initialize OCR engine globally to avoid re-init overhead
ocr_engine_instance = RapidOCR()

# Set tesseract path if needed (Windows usually requires this if not in PATH)
# converting to raw string to avoid escape issues
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_for_ocr(crop):
    """
    Basic preprocessing for OCR: Grayscale -> Threshold
    """
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
        
    # Validating if image is not empty
    if gray.size == 0:
        return gray
        
    # Otsu's thresholding often works best for text on widely varying backgrounds
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text_from_result(image, result, class_names, engine=None):
    """
    Run OCR on all detected boxes in the YOLO result.
    Returns a list of dicts:
    [
        {
            'class_id': int,
            'class_name': str,
            'box': [x1, y1, x2, y2],
            'conf': float,
            'text': str
        }, ...
    ]
    """
    extracted_data = []
    
    # Use global engine if none provided
    if engine is None:
        engine = ocr_engine_instance
    boxes = result.boxes
    if boxes is None:
        return []

    # Sort boxes by center_y, then center_x as requested
    def get_center_key(box):
        xyxy = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2
        return (cy, cx)

    # Convert to list and sort
    sorted_boxes = sorted(boxes, key=get_center_key)
        
    # Iterate through each box
    for i, box in enumerate(sorted_boxes):
        # Coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Valid crop check
        h, w = image.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop = image[y1:y2, x1:x2]
        
        # Preprocess crop
        clean_crop = preprocess_for_ocr(crop)
        
        # Run Tesseract
        # psm 7 = Treat the image as a single text line.
        # Run RapidOCR with use_det=False since we are processing crops
        try:
            # result from RapidOCR(use_det=False, use_rec=True) is typically: ([['text', score]], [time])
            ocr_result = engine(clean_crop, use_det=False, use_cls=False, use_rec=True)
            
            # DEBUG: Print raw result to see what's happening
            # print(f"DEBUG OCR Result: {ocr_result}")

            text = ""
            if ocr_result:
                # 1. Unpack Result List/Tuple
                # RapidOCR returns a tuple (data, time)
                if isinstance(ocr_result, tuple):
                    data = ocr_result[0]
                else:
                    data = ocr_result
                
                # 2. Extract Text from Data List
                # Data is typically [[text, score], [text2, score2]...]
                if isinstance(data, list) and len(data) > 0:
                     first_match = data[0] # ['Text', score]
                     if isinstance(first_match, (list, tuple)) and len(first_match) > 0:
                         text = first_match[0]
                         
        except Exception as e:
            text = ""
            print(f"OCR Error: {e}")
            
        # Get Class info
        cls_id = int(box.cls[0].item())
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        conf = float(box.conf[0].item())
        
        extracted_data.append({
            'class_id': cls_id,
            'class_name': cls_name,
            'box': [x1, y1, x2, y2],
            'conf': conf,
            'text': text
        })
        
    return extracted_data

def group_by_lines(extracted_data, y_tolerance=20):
    """
    Group boxes into lines based on Y-coordinate proximity.
    Assumes extracted_data is already roughly sorted by Y.
    """
    lines = []
    current_line = []
    
    for item in extracted_data:
        box = item['box']
        cy = (box[1] + box[3]) / 2
        
        if not current_line:
            current_line.append(item)
            continue
            
        # Check against average Y of current line
        line_ys = [(i['box'][1] + i['box'][3])/2 for i in current_line]
        avg_y = sum(line_ys) / len(line_ys)
        
        if abs(cy - avg_y) < y_tolerance:
            current_line.append(item)
        else:
            # Sort current line by X before finishing
            current_line.sort(key=lambda x: x['box'][0])
            lines.append(current_line)
            current_line = [item]
            
    # Append last line
    if current_line:
        current_line.sort(key=lambda x: x['box'][0])
        lines.append(current_line)
        
    return lines
