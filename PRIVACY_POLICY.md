# Privacy Policy for Receipt OCR Dashboard

**Effective Date:** 2025-12-26

## 1. Introduction

Receipt OCR Dashboard ("we," "us," or "our") is a personal portfolio project designed to demonstrate optical character recognition (OCR) and large language model (LLM) capabilities on receipt images. This Privacy Policy explains how we collect, use, and disclose information when you use our web application.

## 2. Information We Collect

Because this application is designed to respect your privacy, we collect minimal data:

- **Google Account Information:** We access your Google Drive solely for the purpose of saving the data you explicitly choose to export. We do not store your password or login credentials; authentication is handled securely via Google's OAuth 2.0 service.
- **User Content:** When you upload an image of a receipt, the image is processed in memory by our application to extract text.
- **Cookies:** We use cookies only as necessary for the functioning of the application (e.g., maintaining your session).

## 3. How We Use Your Information

- **Receipt Processing:** Uploaded images are processed to extract text (store name, date, items, prices). This processing happens on our server (Streamlit Cloud).
- **Data Storage:** The extracted data is sent DIRECTLY to your own Google Drive account (specifically `receipts_log.csv`). We do not maintain a separate database of your receipts.
- **AI Processing:** Receipt text may be sent to third-party LLM providers (e.g., OpenRouter, Google Gemini) solely for the purpose of structuring the data. These providers are not permitted to use your data for training their models.

## 4. Data Sharing and Disclosure

We do not sell, trade, or otherwise transfer your personally identifiable information to outside parties. Your data stays between You, the Application Session, and Your Google Drive.

## 5. Security

We implement reasonable security measures to maintain the safety of your personal information. Authentication tokens are stored locally in your browser session and are discarded when you close the tab.

## 6. Access to Data (Google Drive Scope)

Our application requests the `drive.file` scope. This means:

- We can ONLY access files created or opened by this app.
- We CANNOT see, edit, or delete the rest of your Google Drive files.

## 7. Contact Us

If you have any questions about this Privacy Policy, please contact us at:
**Email:** jitwongjit@gmail.com
