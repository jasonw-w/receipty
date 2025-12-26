import streamlit as st
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO
from drive_api import DriveAPI

@st.cache_resource
def load_ocr_engine():
    """Loads RapidOCR once. This is huge for performance."""
    print("Loading OCR Engine into Memory...")
    return RapidOCR()

@st.cache_resource
def load_yolo_model(model_path):
    """Loads YOLO model once."""
    print(f"Loading YOLO from {model_path}...")
    return YOLO(model_path)

@st.cache_resource
def load_drive_api():
    """Authenticates with Google only once."""
    print("Authenticating Google Drive...")
    return DriveAPI()