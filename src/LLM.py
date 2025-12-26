from openrouter import OpenRouter
import os
from dotenv import load_dotenv
import json
# Load env variables from current directory
load_dotenv()

client = OpenRouter(
    api_key=os.environ.get("HACKCLUB_LLM_API_KEY"),
    server_url="https://ai.hackclub.com/proxy/v1",
)
default = str(sorted(json.load(open("src/item_categories.json"))))
def LLM(text, item_categories=default):
    response = client.chat.send(
        model="google/gemini-3-flash-preview",  # Using a strong model for OCR tasks,
        temperature=0.7,
        messages=[
        {
            "role": "system",
            "content": f"""
            You are a receipt parsing assistant. Extract data into a structured JSON format.
            **NOTE**: If there is a discount, apply it to the price of the item. If it is percent off, apply it to the price of all item. If you are not sure about the category, just return "Other".
            
            STRICTLY USE ONLY THE FOLLOWING CATEGORIES for the 'category' field:
            {item_categories}
            
            Return a JSON object with the following schema:
            {{
                "store_name": "string(all in lower case)",
                "date": "string (YYYY-MM-DD)",
                "items": [
                    {{
                        "item_name": "string (original name)",
                        "short_name": "string (concise name)",
                        "category": "string (Selected from the list above)",
                        "price": number,
                        "confidence": boolean
                    }}
                ],
                "total": {{
                    "valid": boolean,
                    "actual_total": number,
                    "calculated_sum": number,
                    "note": "string(dont print anything if note is empty)"
                }}
            }}
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


    