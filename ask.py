from flask import Flask, request, jsonify, Response
import requests
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import re
import json
import os
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "https://id-preview--3474ee39-2650-4791-ae98-e4e2992f0966.lovable.app"
])

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_LLM_MODEL = os.getenv("TOGETHER_LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")


def query_llama(prompt: str) -> str:
    """Queries the Together AI LLM with the given prompt."""
    if not TOGETHER_API_KEY:
        raise EnvironmentError("TOGETHER_API_KEY is not set")

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


def extract_text_from_pdf_text_layer(file_stream) -> str:
    """Extracts text from a searchable PDF."""
    try:
        reader = PdfReader(file_stream)
        return "\n".join(
            f"--- Page {i + 1} ---\n{page.extract_text().strip()}"
            for i, page in enumerate(reader.pages)
            if page.extract_text()
        )
    except Exception:
        return ""


def extract_text_from_scanned_pdf(file_stream) -> str:
    """Performs OCR on scanned PDF pages."""
    try:
        images = convert_from_bytes(file_stream.read())
        return "\n".join(
            f"--- OCR Page {i + 1} ---\n{pytesseract.image_to_string(image).strip()}"
            for i, image in enumerate(images)
        )
    except Exception:
        return ""


def extract_json_block(text: str) -> dict | None:
    """Attempts to extract a JSON object containing a 'fields' array."""
    text = text.strip()
    try:
        if text.startswith("["):
            end_idx = text.find("]") + 1
            parsed_list = json.loads(text[:end_idx])
            return {"fields": parsed_list}

        if '"fields"' in text:
            match = re.search(r'\{\s*"fields"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))

        if '"answer"' in text and '"fields"' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])

        return None
    except Exception:
        return None


def build_auto_field_prompt(document_text: str) -> str:
    return f"""You are a document analysis AI that extracts structured data from a document.

Given the OCR-parsed content of a document, extract all identifiable fields and return them in this JSON format:
[
  {{
    "category": "string",
    "field_name": "string",
    "value": "string",
    "confidence": float,
    "bbox": [x0, y0, x1, y1],
    "page_number": integer
  }}
]

Return only the JSON array.

Document:
{document_text}
"""

def build_targeted_field_prompt(document_text: str, fields_to_extract: list[str]) -> str:
    fields_list_str = ", ".join(f'"{field}"' for field in fields_to_extract)
    return f"""You are a document analysis AI.

Extract the following fields from the document: {fields_list_str}

Return only a JSON array where each object follows this schema:
{{
  "category": "string",
  "field_name": "string",
  "value": "string",
  "confidence": float,
  "bbox": [x0, y0, x1, y1],
  "page_number": integer
}}

Document:
{document_text}
"""


def build_question_with_dimensions_prompt(document_text: str, user_question: str) -> str:
    return f"""You are a document understanding AI.

Given the document content and a user question, return a structured JSON containing your natural language answer and all the fields (with bounding boxes) used in the answer.

Response format:
{{
  "answer": "Your answer here.",
  "fields": [
    {{
      "field_name": "string",
      "value": "string",
      "confidence": float,
      "bbox": [x0, y0, x1, y1],
      "page_number": integer
    }}
  ]
}}

Document:
{document_text}

Question:
{user_question}
"""


@app.route("/auto-extract-fields", methods=["POST"])
def auto_extract_fields():
    """Extract fields from uploaded document using LLM."""
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

        # Return raw_output directly if parsing fails
        return Response(raw_output, mimetype="application/json")

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
        return jsonify({"error": "DOCUMENT_EMPTY_OR_UNREADABLE"}), 400

    prompt = build_question_with_dimensions_prompt(document_text, question)
    try:
        raw_output = query_llama(prompt)
        return jsonify({"raw_output": raw_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/extract-some-fields", methods=["POST"])
def extract_some_fields():
    """Extract only specific fields from a document using LLM."""
    if "file" not in request.files or "fields_to_extract" not in request.form:
        return jsonify({"error": "Missing 'file' or 'fields_to_extract' in form-data"}), 400

    file = request.files["file"]
    fields_raw = request.form["fields_to_extract"]

    try:
        fields_to_extract = json.loads(fields_raw)
        if not isinstance(fields_to_extract, list) or not all(isinstance(f, str) for f in fields_to_extract):
            raise ValueError("Invalid format: fields_to_extract must be a list of strings")
    except Exception:
        return jsonify({"error": "fields_to_extract must be a JSON array of strings"}), 400

    file.seek(0)
    document_text = extract_text_from_pdf_text_layer(file)
    if not document_text.strip():
        file.seek(0)
        document_text = extract_text_from_scanned_pdf(file)
    if not document_text.strip():
        return jsonify({"error": "DOCUMENT_EMPTY_OR_UNREADABLE"}), 400

    prompt = build_targeted_field_prompt(document_text, fields_to_extract)

    try:
        raw_output = query_llama(prompt)
        parsed_json = extract_json_block(raw_output)

        if parsed_json and "fields" in parsed_json:
            return jsonify({"fields_json": parsed_json})

        return Response(raw_output, mimetype="application/json")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
