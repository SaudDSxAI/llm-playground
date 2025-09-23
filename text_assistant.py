import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI  # updated import
from langchain.schema import HumanMessage, SystemMessage

# ================= CONFIG =================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file")

DATA_DIR = Path("data")
COMBINED_FILE = DATA_DIR / "summarize.txt"
PROMPT_FILE = DATA_DIR / "prompt.txt"

# ================= LOAD PROMPT =================
def load_prompt(prompt_file=PROMPT_FILE):
    if not prompt_file.exists():
        raise FileNotFoundError(f"❌ Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    return prompt_text  # return as string

# ================= LOAD TEXT =================
def load_text(file_path=COMBINED_FILE):
    if not file_path.exists():
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"✅ Loaded text from {file_path}")
    return text

# ================= BUILD QA CHAIN =================
def build_qa_chain(context_text, system_prompt_text):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    def qa_chain(query):
        messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=f"Here is the context:\n{context_text}\n\nNow answer the question:\n{query}")
        ]
        response = llm(messages)
        return response

    return qa_chain

# ================= MAIN PIPELINE =================
def run_pipeline():
    print("🔄 Loading recruiter prompt...")
    system_prompt_text = load_prompt()

    print("🔄 Loading combined text...")
    combined_text = load_text()

    print("🔄 Building QA chain...")
    qa_chain = build_qa_chain(combined_text, system_prompt_text)

    print("\n🚀 Text Assistant Ready! Ask me questions (type 'exit' to quit)\n")
    while True:
        query = input("Query: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting assistant. Goodbye!")
            break
        print("💭 Thinking...")
        try:
            result = qa_chain(query)
            print("\nAnswer:", result.content)
        except Exception as e:
            print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    run_pipeline()
