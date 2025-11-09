import os
import pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# UPDATED IMPORTS - Works with latest LangChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set API key
os.environ["PINECONE_API_KEY"] = "pcsk_5CX4Px_31FNGf7oNNNcpn2qVuXJjQXAUjSvvtJAeD4cHNC2hBKVJXE84Qj1bEYJTiAfoyq"

# ==================== DOCUMENT LOADING ====================
def read_doc(directory):
    """Load PDF documents from directory"""
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# ==================== CHUNKING ====================
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(docs)
    return docs

# ==================== EMBEDDINGS ====================
def get_embeddings():
    """Load free HuggingFace embedding model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# ==================== PINECONE STORAGE ====================
def store_embeddings_pinecone(documents, index_name):
    """Store document embeddings in Pinecone"""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Get embeddings
    embeddings = get_embeddings()
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f" Index '{index_name}' not found. Please create it in Pinecone first.")
    
    # Connect to index
    index = pc.Index(index_name)
    
    # Create vectorstore
    vectorstore = PineconeVectorStore.from_documents(
        documents, 
        embeddings, 
        index_name=index_name
    )
    return vectorstore

# ==================== LOAD EXISTING VECTORSTORE ====================
def load_vectorstore(index_name):
    """Load existing Pinecone vectorstore for retrieval"""
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    return vectorstore

# ==================== LLM SETUP ====================
def load_llm(model_name="llama3"):
    """Load local Ollama LLM"""
    return OllamaLLM(model=model_name)

# ==================== FORMAT DOCUMENTS ====================
def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

# ==================== CREATE RAG CHAIN (NEW METHOD) ====================
def create_rag_chain(vectorstore, llm):
    """Create RAG chain using LCEL (LangChain Expression Language)"""
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create prompt template
    template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Create RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# ==================== QUERY WITH SOURCES ====================
def query_with_sources(question, retriever, rag_chain):
    """Query the system and return answer with sources"""
    
    # Get answer
    answer = rag_chain.invoke(question)
    
    # Get source documents - UPDATED METHOD
    source_docs = retriever.invoke(question)
    
    return {
        "answer": answer,
        "source_documents": source_docs
    }

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    
    # STEP 1: Load and process documents (run once)
    print(" Loading documents...")
    doc = read_doc('documents/')
    print(f"✓ Loaded {len(doc)} documents")
    
    # STEP 2: Chunk documents
    print("  Chunking documents...")
    documents = chunk_data(docs=doc)
    print(f"✓ Created {len(documents)} chunks")
    
    # STEP 3: Store embeddings (run once, then comment out)
    index_name = "chatbox"
    print(" Storing embeddings in Pinecone...")
    vectorstore = store_embeddings_pinecone(documents, index_name)
    print("✓ Embeddings created and stored successfully!")
    
    # ==================== RETRIEVAL PART (NEW METHOD) ====================
    
    # STEP 4: Load vectorstore (for subsequent runs, uncomment below)
    # print("\n Loading vectorstore for retrieval...")
    # vectorstore = load_vectorstore(index_name)
    
    # STEP 5: Load LLM
    print(" Loading LLM...")
    llm = load_llm(model_name="llama3")
    
    # STEP 6: Create RAG chain
    print(" Creating RAG chain...")
    rag_chain, retriever = create_rag_chain(vectorstore, llm)
    
    # STEP 7: Test queries
    print("\n" + "="*70)
    print(" RAG SYSTEM READY - Testing queries")
    print("="*70 + "\n")
    
    # Example queries
    test_queries = [
        "What is the main purpose of the document?",
        "Summarize the key points",
        "What are the main topics discussed?"
    ]
    
    for query in test_queries[:1]:  # Test with first query
        print(f"❓ Question: {query}")
        print("-" * 70)
        
        try:
            # Query with sources
            result = query_with_sources(query, retriever, rag_chain)
            
            print("\n Answer:")
            print(result["answer"])
            
            print("\n Sources used:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n--- Source {i} ---")
                print(doc.page_content[:300] + "...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"Metadata: {doc.metadata}")
            
            print("\n" + "="*70 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

# ==================== INTERACTIVE CHAT LOOP ====================
def chat_loop():
    """Interactive chat interface"""
    index_name = "chatbox"
    
    print("🔄 Loading system...")
    # Load vectorstore and create chain
    vectorstore = load_vectorstore(index_name)
    llm = load_llm()
    rag_chain, retriever = create_rag_chain(vectorstore, llm)
    
    print("\n" + "="*70)
    print(" Chatbot ready! Type 'quit' to exit, 'sources' to see last sources")
    print("="*70 + "\n")
    
    last_sources = []
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print(" Goodbye!")
            break
        
        if query.lower() == 'sources' and last_sources:
            print("\n Last query sources:")
            for i, doc in enumerate(last_sources, 1):
                print(f"\n--- Source {i} ---")
                print(doc.page_content[:300] + "...")
            print()
            continue
        
        if not query:
            continue
        
        try:
            result = query_with_sources(query, retriever, rag_chain)
            print(f"\n Bot: {result['answer']}\n")
            last_sources = result['source_documents']
        except Exception as e:
            print(f"Error: {e}\n")

# Uncomment to run interactive chat:
# chat_loop()
