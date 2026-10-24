from rag_chatbot import RAGChatbot
import os

def main():
    # Check if documents folder exists
    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("⚠️  Created 'documents' folder. Please add PDF files there.")
        return
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Setup pipeline
    if chatbot.load_documents() == 0:
        print("❌ No PDF documents found in 'documents' folder!")
        print("📝 Please add your PDFs to: ./documents/")
        return
    
    print("\n⏳ Processing documents... (this may take a few minutes)")
    chunks = chatbot.chunk_documents()
    chatbot.create_vector_store(chunks)
    chatbot.build_qa_chain()
    
    # Interactive chat
    print("\n" + "="*60)
    print("✅ RAG Chatbot Ready!")
    print("="*60)
    print("Type your questions below. Type 'exit' to quit.\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() == 'exit':
                print("Goodbye! 👋")
                break
            
            if not question:
                continue
            
            print("\n🤔 Thinking...")
            answer, sources = chatbot.ask(question)
            print(f"\nBot: {answer}")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()