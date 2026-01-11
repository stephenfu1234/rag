#conda activate rag_env  

import os
import time
from pathlib import Path
from typing import List

# ========================================================
# LANGCHAIN CLASSIC IMPORTS
# ========================================================
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.llms import OpenAI  # only if using old LLMs, but you're using ChatOpenAI
# (You keep ChatOpenAI import below because it's not in langchain_classic)

# ========================================================
# OTHER LANGCHAIN IMPORTS (NO CLASSIC VERSIONS)
# ========================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env

from tqdm import tqdm

# ========================================================
# CONFIG
# ========================================================
PDF_DIR = "word/pdf"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

OPENAI_PRICING = {
    "gpt-4o-mini": {"prompt": 0.0015, "completion": 0.002},
    "text-embedding-3-small": 0.0004
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ========================================================
# HELPERS
# ========================================================
def load_pdfs_from_folder(pdf_folder: str) -> List:
    docs = []
    for p in Path(pdf_folder).glob("**/*.pdf"):
        loader = PyPDFLoader(str(p))
        file_docs = loader.load()
        print(f"Loaded {len(file_docs)} pages from {p.name}")
        docs.extend(file_docs)
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = []
    for d in tqdm(docs, desc="Splitting docs"):
        split_docs.extend(splitter.split_documents([d]))
    print(f"Total chunks after split: {len(split_docs)}")
    return split_docs


def create_or_load_vectorstore(pdf_dir=PDF_DIR, persist_dir=CHROMA_DIR, force_rebuild=False):
    """
    Load an existing Chroma vectorstore OR build a new one.
    When building for the first time, return metadata with estimated cost & time.
    """
    persist_path = Path(persist_dir)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # --- Case 1: Load existing DB ---
    if persist_path.exists() and not force_rebuild:
        print(f"Loading existing Chroma DB from {persist_dir}")
        vectordb = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings
        )
        return vectordb, None  # No metadata when loading

    # --- Case 2: Build new DB ---
    print("Building vectorstore from PDFs...")
    docs = load_pdfs_from_folder(pdf_dir)
    if not docs:
        raise ValueError(f"No PDFs found in {pdf_dir}. Place .pdf files there.")

    chunks = split_documents(docs)

    # --- Cost estimation BEFORE embedding ---
    token_counts = [len(c.page_content.split()) for c in chunks]
    total_tokens = sum(token_counts)
    estimated_cost_usd = (total_tokens / 1000) * OPENAI_PRICING["text-embedding-3-small"]

    # --- Time the embedding process ---
    start_time = time.time()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path)
    )
    vectordb.persist()

    end_time = time.time()
    embedding_time_ms = int((end_time - start_time) * 1000)

    metadata = {
        "chunks_added": len(chunks),
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "embedding_time_ms": embedding_time_ms
    }

    print(f"Chroma DB persisted to {persist_dir}")
    return vectordb, metadata



def incremental_add_pdf(vectordb: Chroma, file_path: str) -> dict:
    """
    Add a PDF to an existing vectorstore incrementally and return
    metadata including estimated cost and embedding time.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # --- Split PDF into chunks ---
    chunks = split_documents(docs)

    # --- Deduplicate by content ---
    existing_texts = {d.page_content for d in vectordb.get_all_documents()}
    new_chunks = [c for c in chunks if c.page_content not in existing_texts]

    # --- Cost Estimation BEFORE embedding ---
    # Rough token estimate: 1 token ≈ 1 word
    token_counts = [len(c.page_content.split()) for c in new_chunks]
    total_tokens = sum(token_counts)

    estimated_cost_usd = (total_tokens / 1000) * OPENAI_PRICING["text-embedding-3-small"]

    # --- Time the actual embedding step ---
    start_time = time.time()

    if new_chunks:
        vectordb.add_documents(new_chunks)
        vectordb.persist()

    end_time = time.time()
    embedding_time_ms = int((end_time - start_time) * 1000)

    return {
        "chunks_added": len(new_chunks),
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "estimated_tokens": total_tokens,
        "embedding_time_ms": embedding_time_ms
    }

def build_conversational_chain(vectordb: Chroma, llm_model_name=LLM_MODEL, temperature=0.0):
    """
    Build a conversational retrieval chain.
    """
    llm = ChatOpenAI(model=llm_model_name, temperature=temperature)

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"     # <-- Required for LangChain Classic
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"     # <-- Also required here
    )

    return chain

# ========================================================
# CLI
# ========================================================
def interactive_cli(chain: ConversationalRetrievalChain, vectordb: Chroma):
    print("\n--- RAG CLI ---")
    print("Type 'exit' to quit or 'add <file>' to add new PDFs.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye")
            break

        elif user_input.lower().startswith("add "):
            file_path = user_input[4:].strip()
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                continue
            result = incremental_add_pdf(vectordb, file_path)
            print(f"Added {result['chunks_embedded']} chunks, using {result['estimated_tokens']} estimted tokens, estimated USD cost ${result['estimated_cost_usd']}, time taken ${result['embedding_time_ms']} ms")

        else:
            start = time.time()
            result = chain({"question": user_input})
            end = time.time()

            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            model_price = OPENAI_PRICING.get(LLM_MODEL, {"prompt": 0, "completion": 0})

            estimate = (
                (prompt_tokens / 1000) * model_price["prompt"]
                + (completion_tokens / 1000) * model_price["completion"]
            )

            print("\nAssistant:", result["answer"])

            # show sources
            print ("\nSources:")
            for i, s in enumerate(result.get("source_documents", [])[:5], 1):
                meta = s.metadata or {}
                src = meta.get("source", "unknown")
                pg = meta.get("page", meta.get("page_number", "n/a"))
                snippet = (s.page_content[:300] + "...") if len(s.page_content) > 300 else s.page_content
                print(f"[{i}] {src} (page {pg}) - {snippet}\n")

            print(f"Cost: {round(estimate, 6)} USD, Time: {round((end-start)*1000,2)} ms")
            print("-" * 60)


# ========================================================
# MAIN
# ========================================================
if __name__ == "__main__":
    vectordb, build_metadata = create_or_load_vectorstore()
    chat_chain = build_conversational_chain(vectordb)

    if build_metadata:
        print("Initial vectorstore build metadata:", build_metadata)

    interactive_cli(chat_chain, vectordb)
