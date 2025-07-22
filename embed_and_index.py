from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Save index
vectorstore.save_local("faiss_index")

print("FAISS index built and saved.")
