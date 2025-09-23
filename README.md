LLM Assistant Comparison

In this repository, I am comparing different LLM techniques for the same problem: building an assistant for my professional data that can speak on my behalf. I am experimenting with multiple methods to see which approach works best.

RAG-based Assistant:
The first version uses Retrieval-Augmented Generation (RAG). The assistant loads documents, creates embeddings, stores them in a vector database, and answers questions using context retrieval. The implementation can be found in rag_assistant.py.

Summarization-based Assistant:
The second assistant does not use embeddings. Instead, it loads the text, summarizes it to reduce length and redundancy, and then feeds the summarized content directly to the LLM for answering questions.

This repo demonstrates the workflow, pros, and cons of each approach for creating a personalized professional assistant.