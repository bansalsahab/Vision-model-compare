import os
import re
import json
import base64
from PIL import Image
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai import Credentials

# Model setup
try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
<<<<<<< HEAD
    project_id = "c07c367a-967b-4dba-b687-bf915836ba46"
=======
    project_id = ""
>>>>>>> f9d061e8f52c082a46050b64910b6423bfc65db3

model_id = "meta-llama/llama-3-2-11b-vision-instruct"
params = TextChatParameters(max_tokens=2000, temperature=0)

# Note: Set your API key in environment variable or update credentials
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=os.environ.get("WATSONX_API_KEY", "")
)

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)

def encode_image_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def extract_invoice_data(image_path):
    encoded_string = encode_image_base64(image_path)
    question = (
        "Extract all relevant product and invoice information from this image and return it as a well-structured JSON object. "
        "Group items by category (based on the product description blocks such as MEN'S 96% COTTON 4% SPANDEX WOVEN PANTS/SHORTS etc.). "
        "Each item should include: style_no, quantity, unit, unit_price_usd, total_cost_usd. "
        "Also include: invoice_total_usd, discount_usd, final_total_usd. Return only the JSON output."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encoded_string,
                    }
                }
            ]
        }
    ]

    response = model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]

def group_images_by_name(folder_path):
    grouped = {}
    pattern = re.compile(r"^(.*)_\d+\.png$", re.IGNORECASE)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            match = pattern.match(filename)
            if match:
                key = match.group(1)
                grouped.setdefault(key, []).append(os.path.join(folder_path, filename))

    for key in grouped:
        def sort_key(x):
            search_result = re.search(r"_(\d+)\.png", x)
            return int(search_result.group(1)) if search_result else 0
        grouped[key].sort(key=sort_key)

    return grouped

def smart_merge(json_list):
    merged = {}

    for data in json_list:
        for key, value in data.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                merged[key].extend(value)
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key].update(value)
            elif merged[key] in ["N/A", "", None]:
                merged[key] = value
    return merged

def main(image_folder, output_folder="json_output"):
    os.makedirs(output_folder, exist_ok=True)
    grouped_images = group_images_by_name(image_folder)

    for group_name, image_paths in grouped_images.items():
        print(f"\n📂 Processing: {group_name} ({len(image_paths)} pages)")
        result_jsons = []

        for img_path in image_paths:
            try:
                print(f"   🖼️ {os.path.basename(img_path)}")
                response_text = extract_invoice_data(img_path)
                print("Raw content from model:\n", repr(response_text))

                # Try to extract JSON from markdown-wrapped content
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if match:
                    cleaned_json_str = match.group(1)
                    print("✅ Found JSON inside markdown code block.")
                else:
                    cleaned_json_str = response_text.strip()
                    print("⚠️ No code block found — using raw content directly.")

                try:
                    parsed_json = json.loads(cleaned_json_str)
                    result_jsons.append(parsed_json)
                    print("✅ JSON parsed successfully.")
                except json.JSONDecodeError as e:
                    print("❌ JSON decode error:", e)

            except Exception as e:
                print(f"   ❌ Failed to process {img_path}: {e}")

        # Save merged result
        if result_jsons:
            merged_result = smart_merge(result_jsons)
            output_path = os.path.join(output_folder, f"{group_name}.json")
            with open(output_path, 'w') as out_file:
                json.dump(merged_result, out_file, indent=2)
            print(f"💾 Saved JSON: {output_path}")
        else:
            print(f"⚠️ No valid JSON output for {group_name} — skipping file.")

if __name__ == "__main__":
<<<<<<< HEAD
    main("converted_pngs") 
=======
    main("converted_pngs") 
>>>>>>> f9d061e8f52c082a46050b64910b6423bfc65db3
