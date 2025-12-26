from src.drive_api import DriveAPI
from src.data_extraction import DataExtractor
extractor = DataExtractor()
import streamlit as st
import cv2
import time
import numpy as np

# Initialize Session State for Drive
if 'drive' not in st.session_state:
    st.session_state.drive = None
# --- Top Bar with Status ---
col1, col2 = st.columns([6, 2])
with col1:
    st.title("Receipt OCR")
with col2:
    # Top Right Status Indicator
    if st.session_state.drive:
        st.success("✅ **Drive Connected**")
    else:
        st.error("🔴 **Drive Disconnected**")

descp = "Upload a receipt image (JPG, PNG, BMP, TIFF, WEBP) to extract structured data including date, store name, items, prices, and categories. The extracted data can be automatically saved to your Google Drive."

def stream_data():
    for word in descp.split(" "):
        yield word + " "
        time.sleep(0.03)

if "intro_shown" not in st.session_state:
    st.write_stream(stream_data)
    st.session_state.intro_shown = True
else:
    st.write(descp)

import uuid


# Connect Button
if st.session_state.drive is None or "drive" not in st.session_state:
    if st.button("Connect to Google Drive"):
        with st.spinner("Authenticating..."):
            try:
                st.session_state.drive = DriveAPI()
                st.success("Connected to Google Drive!")
                st.rerun()
            except Exception as e:
                st.error(f"Connection Failed: {e}")
else:
    st.write("Already connected to Google Drive. yay!")

import pillow_heif
from PIL import Image
import io

# Register the HEIF opener with Pillow
pillow_heif.register_heif_opener()

try:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "heic", "heif"])
except Exception as e:
    st.error(f"File Upload Error: {e}, please make sure you are using a valid image file and try again.")

if st.button("Upload image"):
    if uploaded_file is not None:
        start = time.time()
        
        # Check if HEIC/HEIF
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext in ['heic', 'heif']:
            try:
                # Open with Pillow (via pillow-heif)
                image = Image.open(uploaded_file)
                # Convert to RGB
                image = image.convert("RGB")
                # Convert to Numpy array
                img_array = np.array(image)
                # Convert RGB to BGR for OpenCV
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                st.error(f"Error processing HEIC file: {e}")
                img = None
        else:
            # Standard OpenCV decode
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            
        if img is not None:
            # st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Moved to persistent block
            with st.spinner("Extracting data..."):
                data = extractor.extract_data_from_img(img, Test=False)
            end = time.time()
            st.write(f"Time taken: {end - start}")
            st.session_state.data = data
            st.session_state.img = img

# Persistent Display & Edit Block
if "data" in st.session_state and st.session_state.data is not None:
    data = st.session_state.data # Local alias for convenience
    
    # Show Image if available
    if "img" in st.session_state and st.session_state.img is not None:
         st.image(cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2RGB))

    # Edit Loop
    for i in range(len(data.get('items', []))):
        item = data.get('items', [])[i]
        st.write(f"**Item {i + 1}**")
        st.write(f"Name: {item.get('short_name', '')}")
        st.write(f"Price: {item.get('price', '')}")
        st.write(f"Category: {item.get('category', '')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Delete Item", key=f"del_{i}"):
                del st.session_state.data['items'][i]
                st.rerun()
        
        with st.expander(f"Edit Item {i+1}"):
            new_short_name = st.text_input("Short Name", item.get('short_name', ''), key=f"short_name_{i}")
            new_actual_name = st.text_input("Item Name", item.get('item_name', ''), key=f"name_{i}")
            new_price = st.text_input("Price", str(item.get('price', '')), key=f"price_{i}")
            new_cat = st.text_input("Category", item.get('category', ''), key=f"cat_{i}")
            
            # Update state immediately on change
            if new_short_name != item['short_name']:
                st.session_state.data['items'][i]['short_name'] = new_short_name
            if new_actual_name != item['item_name']:
                st.session_state.data['items'][i]['item_name'] = new_actual_name
            if new_price != str(item['price']):
                st.session_state.data['items'][i]['price'] = new_price
            if new_cat != item['category']:
                    st.session_state.data['items'][i]['category'] = new_cat

    # Upload Section (now inside the persistent data block)
    if "drive" in st.session_state and st.session_state.drive is not None:
        st.divider()
        if st.button("Confirm and upload to Drive"):
            st.info("Uploading to Drive...")
            with st.spinner("Uploading to Drive..."):
                try:
                    # Format Data to CSV: ReceiptID,ItemID,Date,Store,Category,Item,Price
                    receipt_id = str(uuid.uuid4())[:8] # Unique ID for this receipt
                    date = data.get('date', '')
                    store = data.get('store_name', '')
                    items = data.get('items', [])
                
                    csv_content = ""
                    for item in items:
                        item_id = str(uuid.uuid4())[:8] # Unique ID for this item row
                        category = item.get('category', '')
                        actual_name = item.get('item_name', '')
                        short_name = item.get('short_name', '')
                        price = item.get('price', '')
                        # Ensure no commas in fields to break CSV
                        actual_name = str(actual_name).replace(',', ' ')
                        short_name = str(short_name).replace(',', ' ')
                        category = str(category).replace(',', ' ')
                        csv_content += f"{receipt_id},{item_id},{date},{store},{category},{short_name},{actual_name},{price}\n"
                    
                    if csv_content:
                        # Update header to include Category, ShortName, ActualName
                        file_id = st.session_state.drive.create_csv_if_not_exists("receipts_log.csv", "ReceiptID,ItemID,Date,Store,Category,ShortName,ActualName,Price\n")
                        st.session_state.drive.append_to_csv(file_id, csv_content)
                        st.success(f"Data appended to Drive CSV! (Receipt ID: {receipt_id})")
                except Exception as e:
                    st.error(f"Drive Upload Error: {e}")
    else:
        st.warning("Connect to Google Drive to enable uploading.")



