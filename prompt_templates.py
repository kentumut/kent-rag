from langchain.prompts import PromptTemplate

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert assistant who answers only based on the context below.\n"
        "If the answer is not present, say \"I don't know.\"\n\n"
        "The context may be written in first person (I, my, me). The question may refer to the same person using 'he', 'his', etc. Match pronouns logically.\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    ),
)
