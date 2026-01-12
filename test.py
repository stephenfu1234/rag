"""
Modern LangChain Conversational Retrieval (LCEL)
Replaces ConversationalRetrievalChain
"""

# ---------- Imports ----------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ---------- LLM ----------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# ---------- Embeddings ----------
embeddings = OpenAIEmbeddings()

# ---------- Example Data ----------
documents = [
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
    Document(page_content="LCEL is the modern, recommended way to build chains in LangChain."),
    Document(page_content="ConversationalRetrievalChain is deprecated and should not be used."),
    Document(page_content="LangGraph is used for complex agent workflows."),
]

# ---------- Vector Store ----------
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------- Prompt: Question Condensing ----------
condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite the user's question as a standalone question."),
        (
            "human",
            "Chat history:\n{chat_history}\n\nUser question:\n{question}",
        ),
    ]
)

condense_chain = (
    condense_prompt
    | llm
    | StrOutputParser()
)

# ---------- Prompt: Answer Generation ----------
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the provided context."),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ]
)

# ---------- Retrieval Function ----------
def retrieve_context(inputs: dict) -> str:
    docs = retriever.get_relevant_documents(inputs["standalone_question"])
    return "\n\n".join(doc.page_content for doc in docs)

# ---------- Full LCEL Chain ----------
chain = (
    {
        "standalone_question": condense_chain,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"],
    }
    | RunnablePassthrough.assign(
        context=RunnableLambda(retrieve_context)
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ---------- Chat Memory (External) ----------
chat_history = []

# ---------- Chat Loop ----------
def chat():
    print("LCEL Conversational RAG (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() in {"exit", "quit"}:
            break

        response = chain.invoke(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )

        print(f"AI: {response}\n")

        # Manual memory update
        chat_history.append(("human", question))
        chat_history.append(("ai", response))


if __name__ == "__main__":
    chat()
