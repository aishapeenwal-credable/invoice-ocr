from flask import Flask, request, jsonify
import requests
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import re
import json
import io
import ast
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_LLM_MODEL = os.getenv("TOGETHER_LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")


def query_llama(prompt):
    if not TOGETHER_API_KEY:
        raise EnvironmentError("TOGETHER_API_KEY is not set in environment variables")

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": TOGETHER_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2000
    }

    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload, verify=False)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def extract_text_from_pdf_text_layer(file_stream):
    try:
        reader = PdfReader(file_stream)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i + 1} ---\n{page_text.strip()}\n"
        return text.strip()
    except Exception as e:
        return ""


def extract_text_from_scanned_pdf(file_stream):
    try:
        images = convert_from_bytes(file_stream.read())
        text = ""
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            text += f"\n--- OCR Page {i + 1} ---\n{page_text.strip()}\n"
        return text.strip()
    except Exception as e:
        return ""


def extract_json_block(text):
    try:
        text = text.strip()

        # If it's a valid JSON array followed by extra non-JSON notes
        if text.startswith("["):
            end_idx = text.find("]") + 1
            try:
                parsed_list = json.loads(text[:end_idx])
                return {"fields": parsed_list}
            except Exception as e:
                print(f"[DEBUG] Failed list parse: {e}")

        # If it's a wrapped fields object
        if '"fields"' in text:
            match = re.search(r'\{\s*"fields"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))

        # If it's a complete object with "answer" and "fields"
        if '"answer"' in text and '"fields"' in text:
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                return json.loads(text[start:end])
            except Exception as e:
                print(f"[DEBUG] Full object parse failed: {e}")

        raise ValueError("No valid JSON block found.")
    except Exception as e:
        print(f"[ERROR] extract_json_block failed: {e}")
        return None


def build_auto_field_prompt(document_text):
    return f"""You are a document analysis AI that extracts structured data from any document.

Given the OCR-parsed content of a document, extract all identifiable fields and return in JSON using this schema:
- category: string
- field_name: string
- value: string
- confidence: float (0.0 to 1.0)
- bbox: [x0, y0, x1, y1]
- page_number: integer

Return only the JSON.

Document:
{document_text}
"""


def build_question_with_dimensions_prompt(document_text, user_question):
    return f"""You are a document understanding AI.

Given the following document and user question, return a strcuture output that contains all relevant extracted fields used in your answer.

Each field should include:
- field_name: string
- value: string
- confidence: float (0.0 to 1.0)
- bbox: [x0, y0, x1, y1]  # in pixels
- page_number: integer

Document:
{document_text}

Question:
{user_question}

Only return the JSON object in this format for /auto-extract-fields
{{
  "answer": "...answer in natural language...",
  "fields": [
    {{
      "field_name": "invoice_number",
      "value": "INV-00123",
      "confidence": 0.95,
      "bbox": [120, 200, 300, 220],
      "page_number": 1
    }}
  ]
  Keep itconversation for /ask endpoint
}}"""


@app.route("/auto-extract-fields", methods=["POST"])
def auto_extract_fields():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in form-data"}), 400

    file = request.files["file"]
    file.seek(0)
    document_text = extract_text_from_pdf_text_layer(file)
    if not document_text.strip():
        file.seek(0)
        document_text = extract_text_from_scanned_pdf(file)
    if not document_text.strip():
        return jsonify({"error": "DOCUMENT_EMPTY_OR_UNREADABLE"}), 400

    prompt = build_auto_field_prompt(document_text)
    try:
        raw_output = query_llama(prompt)
        parsed_json = extract_json_block(raw_output)
        if parsed_json and "fields" in parsed_json:
            return jsonify({"fields_json": parsed_json})
        else:
            return jsonify({"error": "FAILED_TO_PARSE_JSON", "raw_output": raw_output}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    if "file" not in request.files or "question" not in request.form:
        return jsonify({"error": "Missing 'file' or 'question' in form-data"}), 400

    file = request.files["file"]
    question = request.form["question"]
    file.seek(0)
    document_text = extract_text_from_pdf_text_layer(file)
    if not document_text.strip():
        file.seek(0)
        document_text = extract_text_from_scanned_pdf(file)
    if not document_text.strip():
        return jsonify({"error": "DOCUMENT_EMPTY_OR_UNREADABLE"})

    prompt = build_question_with_dimensions_prompt(document_text, question)
    try:
        raw_output = query_llama(prompt)
        return jsonify({"raw_output": raw_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)