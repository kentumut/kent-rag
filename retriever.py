from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

