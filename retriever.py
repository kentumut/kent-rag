from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

INDEX_PATH = "faiss_index"

def build_faiss_index():
    docs = []
    # Load your docs, adjust paths as needed
    docs.extend(TextLoader("data/about_me.txt").load())
    docs.extend(TextLoader("data/resume.txt").load())
    docs.extend(TextLoader("data/faq.txt").load())
    # Add more loaders if you want

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(INDEX_PATH)
    print("FAISS index built and saved.")

def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vectorstore = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever
