import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# ================= CONFIG =================
load_dotenv()
HF_ACCESS_TOKEN = os.getenv("HF_Access_Token")
if not HF_ACCESS_TOKEN:
    raise ValueError("❌ Missing HF_Access_Token in .env file")

DATA_DIR = Path("data")
COMBINED_FILE = DATA_DIR / "summarize.txt"  # context text (optional)
PROMPT_FILE = DATA_DIR / "prompt.txt"       # system prompt

# Initialize Hugging Face OpenAI-compatible client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_ACCESS_TOKEN,
    )

# ================= LOAD PROMPT =================
def load_prompt(prompt_file=PROMPT_FILE):
    if not prompt_file.exists():
        raise FileNotFoundError(f"❌ Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    return prompt_text

# ================= LOAD TEXT =================
def load_text(file_path=COMBINED_FILE):
    if not file_path.exists():
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"✅ Loaded text from {file_path}")
    return text

# ================= BUILD CHAT/QA CHAIN =================
def build_qa_chain(context_text, system_prompt_text):
    conversation_history = [
        {"role": "system", "content": system_prompt_text}
    ]

    def qa_chain(query):
        # Add context and user query
        conversation_history.append(
            {"role": "user", "content": f"Here is the context:\n{context_text}\n\nQuestion:\n{query}"}
        )

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:fireworks-ai",
            messages=conversation_history
        )

        assistant_reply = response.choices[0].message.content
        # Add assistant reply to history for context retention
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    return qa_chain

# ================= MAIN PIPELINE =================
def run_pipeline():
    print("🔄 Loading system prompt...")
    system_prompt_text = load_prompt()

    print("🔄 Loading combined context text...")
    combined_text = load_text()

    print("🔄 Building QA/chat chain...")
    qa_chain = build_qa_chain(combined_text, system_prompt_text)

    print("\n🚀 Llama Chatbot Ready! Ask questions (type 'exit' to quit)\n")
    while True:
        query = input("Query: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting chatbot. Goodbye!")
            break
        print("💭 Thinking...")
        try:
            answer = qa_chain(query)
            print("\nAnswer:", answer, "\n")
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    run_pipeline()