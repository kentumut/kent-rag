from rag_chain import query_rag

if __name__ == "__main__":
    print("Type 'exit' to quit.")
    while True:
        user_input = input("Ask me anything about Kent > ")
        if user_input.lower() in {"exit", "quit"}:
            break
        response = query_rag(user_input)
        print("\nAnswer:", response["answer"])
        print("Sources:", response["sources"])
        print("-" * 40)
