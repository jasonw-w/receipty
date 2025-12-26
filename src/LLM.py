from openrouter import OpenRouter
import os
from dotenv import load_dotenv

# Load env variables from current directory
load_dotenv()

client = OpenRouter(
    api_key=os.environ.get("HACKCLUB_LLM_API_KEY"),
    server_url="https://ai.hackclub.com/proxy/v1",
)

def LLM(text):
    response = client.chat.send(
        model="google/gemini-3-flash-preview",  # Using a strong model for OCR tasks,
        temperature=0.7,
        messages=[
        {
            "role": "system",
            "content": """
            You are a receipt parsing assistant. Extract data into a structured JSON format.
            
            Return a JSON object with the following schema:
            {
                "store_name": "string",
                "date": "string (YYYY-MM-DD)",
                "items": [
                    {
                        "item_name": "string (original name)",
                        "short_name": "string (concise name)",
                        "category": "string (Food, Groceries, etc.)",
                        "price": number,
                        "confidence": boolean
                    }
                ],
                "total": {
                    "valid": boolean,
                    "actual_total": number,
                    "calculated_sum": number,
                    "note": "string"
                }
            }
            OUTPUT ONLY VALID JSON. NO MARKDOWN. NO CODE BLOCKS.
            """
        },
        {
            "role": "user",
            "content": text
        },
        ],
        stream=False,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Test the function
    try:
        response_text = LLM("test")
        print("Response:")
        print(response_text)
    except Exception as e:
        print(f"Error: {e}")
