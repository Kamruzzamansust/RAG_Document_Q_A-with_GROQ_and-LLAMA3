import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=.7)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    {context}
    <context>

    Question: {input}
    """
)

def create_vector_embedding():
    if "vaector" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader =PyPDFLoader("D:\All_data_science_project\Langchain\RAG_Document_Q_A with_GROQ_and LLAMA3\DATA\Power bi 0 To Dax .pdf")
        try:
            st.session_state.docs = st.session_state.loader.load()
        except Exception as e:
            st.error(f"Error loading PDFs: {e}")
            return
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

user_prompt = st.text_input("Enter your query from the books")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write('Your Vector Database is Ready Now !!!!!!!!!')

import time
if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start_time = time.process_time()
    try:
        response = retrieval_chain.invoke({'input': user_prompt})
        print(f'Response time :{time.process_time()-start_time}')
        st.write(response['answer'])
    except Exception as e:
        st.error(f"Error processing query: {e}")

    with st.expander('Document Simililarity Search'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-----------------')
