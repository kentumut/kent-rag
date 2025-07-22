from mistral_llm_wrapper import MistralChat
from retriever import load_retriever
from prompt_templates import rag_prompt
from langchain.chains import RetrievalQA

def create_rag_chain():
    retriever = load_retriever()
    llm = MistralChat()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt},
    )
    return qa_chain

qa_chain = create_rag_chain()

def query_rag(question: str):
    result = qa_chain(question)
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source", "unknown") for doc in result["source_documents"]],
    }
