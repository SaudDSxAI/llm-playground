import ollama_test

# --- Configuration ---
MODEL = "llama2"
SYSTEM_PROMPT = (
    "You are Saud's helpful and forward-thinking AI assistant. "
    "You always give insightful, concise answers and maintain context. "
    "When unsure, you reason carefully instead of guessing."
)

def chat():
    print("💬 Saud's Assistant (type 'exit' to quit)\n")
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Add user message
        history.append({"role": "user", "content": user_input})

        # Generate response from Ollama
        response = ollama_test.chat(model=MODEL, messages=history)

        # Extract model reply
        reply = response["message"]["content"].strip()
        print(f"Assistant: {reply}\n")

        # Keep conversation history
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    chat()
