import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= CONFIG =================
load_dotenv()  # Load environment variables from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file")

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "github_data.txt"
OUTPUT_FILE = DATA_DIR / "summarize.txt"
MAX_WORKERS = 5

# ================= LOAD DATA =================
def load_data(file_path=INPUT_FILE):
    if not file_path.exists():
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"✅ Loaded data from {file_path}")
    return text

# ================= SPLIT INTO CHUNKS =================
def split_text(text, chunk_size=1200, chunk_overlap=0):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents([Document(page_content=text)])
    print(f"✅ Split text into {len(docs)} chunks")
    return docs

# ================= SUMMARIZE CHUNKS IN PARALLEL =================
def summarize_chunk(llm, doc: Document, idx: int):
    print(f"💭 Summarizing chunk {idx + 1}...")
    resp = llm.invoke([
        {"role": "system", "content": "Extract structured notes. Keep all important details."},
        {"role": "user", "content": doc.page_content}
    ])
    print(f"✅ Finished chunk {idx + 1}")
    return resp.content

def summarize_chunks_parallel(docs):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(summarize_chunk, llm, doc, idx): idx for idx, doc in enumerate(docs)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                idx = futures[future]
                print(f"⚠️ Error summarizing chunk {idx + 1}: {e}")

    return results

# ================= HIERARCHICAL FINAL SUMMARY =================
def merge_partial_summaries(partials):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    print("🔄 Merging partial summaries into final summary...")
    final = llm.invoke([
        {"role": "system", "content": "Merge the following summaries into one concise, structured summary without repetition. Use sections: Profile, Skills, Projects, Education, Soft Skills."},
        {"role": "user", "content": "\n\n".join(partials)}
    ])
    print("✅ Final summary created")
    return final.content

# ================= SAVE SUMMARY =================
def save_summary(summary, output_file=OUTPUT_FILE):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✅ Summary saved → {output_file}")

# ================= PIPELINE =================
def run_pipeline():
    text = load_data()
    chunks = split_text(text, chunk_size=1200, chunk_overlap=0)  # minimal overlap
    partial_summaries = summarize_chunks_parallel(chunks)
    final_summary = merge_partial_summaries(partial_summaries)
    save_summary(final_summary)

if __name__ == "__main__":
    run_pipeline()
