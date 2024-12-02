
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# # Load environment variables
# load_dotenv()

# # Initialize Google AI components
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
# embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # Load and process documents
# try:
#     loader = TextLoader("docs/FAQs.txt")
#     docs = loader.load()
# except Exception as e:
#     print(f"Error while loading file: {e}")
#     docs = []

# # Split documents
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# split_docs = text_splitter.split_documents(docs)

# # Create vector store
# vectorstore = FAISS.from_documents(split_docs, embedding)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # Define a prompt template
# prompt_template = """Use the following context to answer the question. 
# If the answer is not in the context, say "I don't have information about that in the current document."

# Context: {context}

# Question: {question}

# Helpful Answer:"""

# prompt = PromptTemplate.from_template(prompt_template)

# # Create a chain that retrieves context, formats the prompt, and generates an answer
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Interactive chat loop
# def main():
#     print("RAG Chatbot is ready! Type 'exit' to quit.")
#     while True:
#         human_message = input("How can I help you today? ")
        
#         # Check for exit condition
#         if human_message.lower() == 'exit':
#             print("Goodbye!")
#             break
        
#         try:
#             # Generate response using the RAG chain
#             response = rag_chain.invoke(human_message)
#             print("\nBot:", response)
#         except Exception as e:
#             print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

def initialize_rag_chain():
    """Initialize the RAG chain components."""
    # Initialize Google AI components
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load and process documents
    try:
        loader = TextLoader("docs/FAQs.txt")
        docs = loader.load()
    except Exception as e:
        st.error(f"Error while loading file: {e}")
        return None

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Create vector store
    vectorstore = FAISS.from_documents(split_docs, embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define a prompt template
    prompt_template = """Use the following context to answer the question. 
    If the answer is not in the context, say "I don't have information about that in the current document."

    Context: {context}

    Question: {question}

    Helpful Answer:"""

    prompt = PromptTemplate.from_template(prompt_template)

    # Create formatting function for documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def main():
    # Set page configuration
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬")

    # Title
    st.title("📖 RAG Chatbot")
    st.write("Ask questions about the content of your document!")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize RAG chain
    rag_chain = initialize_rag_chain()
    if rag_chain is None:
        st.error("Failed to initialize RAG chain. Please check your document and settings.")
        return

    # Chat input
    user_question = st.chat_input("Ask a question about the document")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user question
    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Generate response
        try:
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = rag_chain.invoke(user_question)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()