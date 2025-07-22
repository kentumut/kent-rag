from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Load your documents
docs = []
for filename in os.listdir("data"):
    loader = TextLoader(f"data/{filename}", encoding="utf-8", autodetect_encoding=True)
    docs.extend(loader.load())

# Split them
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed them
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save index
vectorstore.save_local("faiss_index")
