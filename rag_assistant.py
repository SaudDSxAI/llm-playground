import os
from dotenv import load_dotenv
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ================= CONFIG =================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OUTPUT_DIR = Path("data")
COMBINED_FILE = OUTPUT_DIR / "combined.txt"
FAISS_PATH = OUTPUT_DIR / "faiss_index"
PROMPT_FILE = OUTPUT_DIR / "prompt.txt"

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= STEP 0: Load Prompt =================
def load_prompt(prompt_file=PROMPT_FILE):
    if not prompt_file.exists():
        raise FileNotFoundError(f"❌ Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    system_message = SystemMessagePromptTemplate.from_template(
        f"You are Saud's AI Career Assistant.\nAlways follow these recruiter-oriented instructions:\n\n{prompt_text}"
    )
    return system_message

# ================= STEP 1: Load Documents =================
def load_documents(data_folder="data"):
    documents = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    docs = PyPDFLoader(path).load()
                elif file.endswith(".txt"):
                    docs = TextLoader(path, encoding="utf-8").load()
                elif file.endswith(".docx"):
                    docs = UnstructuredWordDocumentLoader(path).load()
                else:
                    continue
                documents.extend(docs)
            except Exception as e:
                print(f"⚠️ Skipping {path}: {e}")
    return [d.page_content for d in documents]

# ================= STEP 2: Combine and Save =================
def combine_and_save(docs, output_file=COMBINED_FILE):
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n\n")
    print(f"✅ Combined data saved → {output_file}")

# ================= STEP 3: Build FAISS =================
def build_vectorstore(combined_file=COMBINED_FILE, faiss_path=FAISS_PATH):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large")
    print("🆕 Creating FAISS index from combined data...")
    with open(combined_file, "r", encoding="utf-8") as f:
        text = f.read()
    vectorstore = FAISS.from_texts([text], embeddings)
    vectorstore.save_local(str(faiss_path))
    return vectorstore

# ================= STEP 4: Build QA Chain =================
def build_qa_chain(vectorstore, system_prompt):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    human_message = HumanMessagePromptTemplate.from_template(
        "Here is the context:\n{context}\n\nNow answer the question:\n{question}"
    )
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

# ================= MAIN PIPELINE =================
def run_pipeline():
    print("🔄 Loading recruiter prompt...")
    system_prompt = load_prompt()

    print("🔄 Loading documents...")
    docs = load_documents()

    print("🔄 Saving combined data...")
    combine_and_save(docs)

    print("🔄 Building vectorstore...")
    vectorstore = build_vectorstore()

    print("🔄 Building QA chain...")
    qa_chain = build_qa_chain(vectorstore, system_prompt)

    print("\n🚀 CV Assistant Ready! Ask me questions (type 'exit' to quit)\n")
    while True:
        query = input("Query: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting assistant. Goodbye!")
            break
        print("💭 Thinking...")
        try:
            result = qa_chain.invoke({"query": query})
            print("\nAnswer:", result["result"])
            print("\nSources:")
            for doc in result["source_documents"]:
                print(" -", doc.metadata.get("source", "unknown"))
        except Exception as e:
            print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    run_pipeline()
