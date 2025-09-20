import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from PyPDF2 import PdfReader
import docx
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# -------------------------------
# Load OpenAI API Key
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# File paths
# -------------------------------
DATA_DIR = Path("data")   # folder containing your .txt, .pdf, .docx
OUTPUT_PATH = Path("data/qa_pairs.jsonl")

# -------------------------------
# Function: Load text from TXT/PDF/DOCX
# -------------------------------
def load_text_from_file(file_path: Path) -> str:
    """
    Loads text from .txt, .pdf, or .docx files.
    """
    ext = file_path.suffix.lower()
    text = ""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
    else:
        print(f"⚠️ Skipping unsupported file type: {file_path}")

    return text.strip()

# -------------------------------
# Function: Split text into overlapping chunks
# -------------------------------
def chunk_text_with_overlap(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Splits text into word chunks with overlap to preserve context.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# -------------------------------
# Function: Generate Q&A from a chunk
# -------------------------------
def generate_qa_from_chunk(chunk: str, num_pairs: int = 3) -> list:
    """
    Uses GPT to generate Q&A pairs (Saud's CV + GitHub context).
    Returns fine-tuning-ready JSONL objects.
    """
    prompt = f"""
You are preparing fine-tuning data for a chatbot that answers questions about Saud's CV and GitHub repositories.

Guidelines:
- Generate {num_pairs} question-answer pairs ONLY about Saud's skills, education, projects, or repositories.
- Each pair must be factual and strictly based on the provided text. Do not hallucinate.
- Keep questions natural (as a user would ask).
- Keep answers concise but informative.
- If text does not provide enough info, skip instead of guessing.
- Format:
Q: <question>
A: <answer>

Text:
{chunk}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise assistant that only generates factual Q&A pairs from Saud's resume and GitHub repositories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
    except Exception as e:
        print(f"❌ API error: {e}")
        return []

    content = response.choices[0].message.content.strip()
    qa_list = []

    # Parse GPT output into Q&A pairs
    lines = content.split("\n")
    current_q, current_a = None, None
    for line in lines:
        if line.startswith("Q:"):
            current_q = line[2:].strip()
        elif line.startswith("A:") and current_q:
            current_a = line[2:].strip()
            qa_list.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions about Saud's resume and GitHub projects."},
                    {"role": "user", "content": current_q},
                    {"role": "assistant", "content": current_a}
                ]
            })
            current_q, current_a = None, None
    return qa_list

# -------------------------------
# Function: Process chunks in parallel
# -------------------------------
def process_chunks_parallel(chunks: list, max_workers: int = 5) -> list:
    """
    Generates Q&A for chunks in parallel and removes duplicate questions.
    """
    all_qa = []
    seen_questions = set()

    def process_single_chunk(idx, chunk):
        print(f"⚡ Submitting chunk {idx+1}/{len(chunks)}...")
        return generate_qa_from_chunk(chunk)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                qa_pairs = future.result()
                for qa in qa_pairs:
                    # Extract user question text
                    user_msg = next((m["content"] for m in qa["messages"] if m["role"] == "user"), "").strip().lower()
                    if user_msg and user_msg not in seen_questions:
                        all_qa.append(qa)
                        seen_questions.add(user_msg)
            except Exception as e:
                print(f"❌ Error processing chunk {idx}: {e}")

    return all_qa

# -------------------------------
# Function: Save Q&A to JSONL
# -------------------------------
def save_qa_to_jsonl(qa_list: list, file_path: Path):
    """
    Saves Q&A pairs in JSONL format for fine-tuning.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for qa in qa_list:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(qa_list)} unique Q&A pairs to {file_path}")

# -------------------------------
# Main Pipeline
# -------------------------------
def main():
    all_text = ""

    # Load all files in data folder
    for file in DATA_DIR.iterdir():
        if file.is_file():
            print(f"📂 Loading {file} ...")
            file_text = load_text_from_file(file)
            if file_text:
                all_text += file_text + "\n"

    if not all_text.strip():
        print("⚠️ No text found in data folder.")
        return

    # Split text into chunks
    chunks = chunk_text_with_overlap(all_text, chunk_size=300, overlap=50)

    # Generate Q&A in parallel
    all_qa = process_chunks_parallel(chunks, max_workers=5)

    # Save final dataset
    save_qa_to_jsonl(all_qa, OUTPUT_PATH)

if __name__ == "__main__":
    main()
