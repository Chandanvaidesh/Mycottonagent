from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
import streamlit as st
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from langdetect import detect

st.title("🌱 Cotton Farming AI Assistant")


# 1. Setup Qdrant + embeddings

client = QdrantClient(host="localhost", port=6333)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Multiple vector stores
cotton_vectorstore = QdrantVectorStore(client=client, collection_name="cotton_chunks", embedding=embeddings)
pest_vectorstore = QdrantVectorStore(client=client, collection_name="cotton_pest_chunks", embedding=embeddings)
scheme_vectorstore = QdrantVectorStore(client=client, collection_name="cotton_scheme_chunks", embedding=embeddings)
rate_vectorstore = QdrantVectorStore(client=client, collection_name="cotton_rate_chunks", embedding=embeddings)

cotton_retriever = cotton_vectorstore.as_retriever(search_kwargs={"k": 5})
pest_retriever = pest_vectorstore.as_retriever(search_kwargs={"k": 5})
scheme_retriever = scheme_vectorstore.as_retriever(search_kwargs={"k": 5})
rate_retriever = rate_vectorstore.as_retriever(search_kwargs={"k": 5})


# 2. Setup OpenRouter LLM

llm = ChatOpenAI(
    api_key="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", 
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-0528:free",
    temperature=0
)


# 3. Prompt template for QA

prompt = ChatPromptTemplate.from_template("""
You are an agricultural AI assistant specialized in cotton farming.
Use the provided context to answer the farmer's query clearly.

Conversation so far:
{chat_history}

<context>
{context}
</context>

Question: {input}
""")


# 4. Memory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# 5. Romanized language handling

def roman_to_native(text: str):
    try:
        lang = detect(text)  # detect language
        if lang == "hi":
            return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        elif lang == "kn":
            return transliterate(text, sanscript.ITRANS, sanscript.KANNADA)
        else:
            return text
    except:
        return text  # fallback if detection fails

# 6. Input processing

def get_inputs(user_input: str):
    # Convert Romanized input to native script
    processed_input = roman_to_native(user_input)

    # Retrieve docs
    cotton_docs = cotton_retriever.get_relevant_documents(processed_input)
    pest_docs = pest_retriever.get_relevant_documents(processed_input)
    scheme_docs = scheme_retriever.get_relevant_documents(processed_input)
    rate_docs = rate_retriever.get_relevant_documents(processed_input)

    all_docs = cotton_docs + pest_docs + scheme_docs + rate_docs
    context_text = "\n\n".join([doc.page_content for doc in all_docs])

    # Format chat history
    memory_vars = memory.load_memory_variables({})
    chat_history = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in memory_vars.get("chat_history", [])]
    )

    return {
        "input": processed_input,
        "chat_history": chat_history,
        "context": context_text
    }

qa_chain = (
    RunnableLambda(get_inputs)
    | prompt
    | llm
    | StrOutputParser()
)


# 7. Streamlit Chat UI

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = qa_chain

# Display past messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input field
if user_query := st.chat_input("Ask about cotton farming..."):
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run QA chain
    response = st.session_state.qa_chain.invoke(user_query)
    memory.save_context({"input": user_query}, {"output": response})

    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
