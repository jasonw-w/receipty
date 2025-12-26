import streamlit as st
import cv2
import time
import numpy as np
import uuid
import pillow_heif
from PIL import Image
import io
from src.drive_api import DriveAPI
from src.data_extraction import DataExtractor
import matplotlib.pyplot as plt
import csv

# Page Config
st.set_page_config(page_title="Receipt OCR Dashboard", layout="wide")

# Initialize Extractor
if 'extractor' not in st.session_state:
    st.session_state.extractor = DataExtractor()

# Register HEIF opener
pillow_heif.register_heif_opener()

# --- Helper Functions ---
def resize_for_display(img, max_width=800):
    """Resize image for faster display in Streamlit"""
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        return cv2.resize(img, (max_width, new_h))
    return img

def stream_data():
    txt = "Upload a receipt image (JPG, PNG, BMP, TIFF, WEBP, HEIC) to extract structured data."
    for word in txt.split(" "):
        yield word + " "
        time.sleep(0.02)

# --- Session State Init ---
if 'drive' not in st.session_state:
    st.session_state.drive = None

# --- Top Bar with Status ---
# --- Top Bar with Status ---
col1, col2 = st.columns([6, 2])
with col1:
    st.title("Receipt OCR Dashboard")
with col2:
    # Top Right Status Indicator
    if st.session_state.drive and st.session_state.drive.service:
        st.success("✅ **Drive Connected**")
    else:
        st.error("🔴 **Drive Disconnected**")

# --- Authentication Logic ---
if st.session_state.drive is None:
    st.session_state.drive = DriveAPI()

# Force-refresh if the object is stale (missing new methods due to caching/cloud state)
if not hasattr(st.session_state.drive, "authenticate_with_code"):
    st.session_state.drive = DriveAPI()
    st.rerun()

# Check for Auth Code in URL (Populated by Google Redirect)
if "code" in st.query_params:
    code = st.query_params["code"]
    # We need to know the Redirect URI that was used.
    # We can try to guess or store it in session state?
    # For now, let's assume the user sets it or we use a standard one.
    redirect_uri = st.session_state.get("redirect_uri", "http://localhost:8501")
    
    with st.spinner("Authenticating with Google..."):
        try:
            st.session_state.drive.authenticate_with_code(code, redirect_uri)
            st.success("Successfully connected!")
            # Clear params to avoid loop
            st.query_params.clear()
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            st.write(f"Ensure the Redirect URI '{redirect_uri}' matches what you configured in Google Cloud Console.")

if not st.session_state.drive.service:
    st.divider()
    st.write("### Connect to Google Drive")
    st.info("Google has blocked the copy-paste flow. You must use the Redirect method.")
    
    # Allow user to set their deployment URL
    redirect_uri = st.text_input("Your App URL (Redirect URI), use http://localhost:8501 for local development", value="https://receipty.streamlit.app", help="Enter the exact URL of this app. Add this to your Google Cloud Console 'Authorized Redirect URIs'.")
    st.session_state["redirect_uri"] = redirect_uri # Store for the callback
    
    try:
        auth_url = st.session_state.drive.get_auth_url(redirect_uri)
        st.link_button("Login with Google", auth_url)
    except Exception as e:
        st.error(f"Could not generate auth URL: {e}")

    st.divider()
st.divider()
c1, c2, c3, c4 = st.columns(4)

# Default values
total_spend_val = "$0.00"
total_items_val = "0"
total_receipts_val = "0"
total_stores_val = "0"

# Try to calculate real metrics if data exists
try:
    if st.session_state.drive:
        file_id_log = st.session_state.drive.get_file_id("receipts_log.csv")
        if file_id_log:
            csv_data_log = st.session_state.drive.read_csv(file_id_log)
            import pandas as pd
            df_log = pd.read_csv(io.StringIO(csv_data_log))
            
            if not df_log.empty and 'Price' in df_log.columns:
                 # Clean Price
                df_log['PriceNumeric'] = df_log['Price'].astype(str).str.replace(r'[$,]', '', regex=True)
                df_log['PriceNumeric'] = pd.to_numeric(df_log['PriceNumeric'], errors='coerce').fillna(0)
                
                total_spend_val = f"${df_log['PriceNumeric'].sum():.2f}"
                total_items_val = str(len(df_log))
                total_receipts_val = str(df_log['ReceiptID'].nunique()) if 'ReceiptID' in df_log.columns else "N/A"
                total_stores_val = str(df_log['Store'].nunique()) if 'Store' in df_log.columns else "N/A"
except Exception:
    pass

with c1:
    st.metric("Total Spend", total_spend_val)
with c2:
    st.metric("Total Items", total_items_val)
with c3:
    st.metric("Total Receipts", total_receipts_val)
with c4:
    st.metric("Total Stores", total_stores_val)
st.divider()

try:
    # Use existing drive instance
    drive = st.session_state.drive
    # Find the correct file ID
    file_id = drive.get_file_id("receipts_log.csv")
    
    if file_id:
        # Read raw CSV string
        csv_data = drive.read_csv(file_id)
        
        # Load into Pandas
        import pandas as pd
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Check if dataframe is not empty
        if not df.empty and 'Price' in df.columns and 'Date' in df.columns:
            # Clean Price Column (Remove '$', ',' and whitespace)
            df['PriceNumeric'] = df['Price'].astype(str).str.replace(r'[$,]', '', regex=True)
            df['PriceNumeric'] = pd.to_numeric(df['PriceNumeric'], errors='coerce').fillna(0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Chart 1: Spend by Date (Interactive Bar Chart)
                st.subheader("Spending by Date")
                daily_spend = df.groupby('Date')['PriceNumeric'].sum()
                st.bar_chart(daily_spend)

            with col2:
                # Chart 2: Spend by Category (Donut Chart using Altair for interactivity)
                st.subheader("Spending by Category")
                if 'Category' in df.columns:
                    import altair as alt
                    cat_spend = df.groupby('Category')['PriceNumeric'].sum().reset_index()
                    
                    chart = alt.Chart(cat_spend).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="PriceNumeric", type="quantitative"),
                        color=alt.Color(field="Category", type="nominal"),
                        tooltip=['Category', 'PriceNumeric']
                    )
                    st.altair_chart(chart, use_container_width=True)

            # Show Data Table
            with st.expander("View Raw Log Data"):
                st.dataframe(df)

        else:
            st.info("Log file is empty or missing columns.")
            if not df.empty: st.dataframe(df)
    else:
        st.info("No 'receipts_log.csv' found in Drive yet.")
        
except Exception as e:
    st.error(f"Error loading Drive data: {e}")
