import openai
import os
import langchain
import pinecone
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.documents import Document   
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain.chains.retrieval import RetrievalQA

os.environ["PINECONE_API_KEY"] = "pcsk_5CX4Px_31FNGf7oNNNcpn2qVuXJjQXAUjSvvtJAeD4cHNC2hBKVJXE84Qj1bEYJTiAfoyq"



def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

doc=read_doc('documents/')
doc
#divide the docs into chunks 
def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs=text_splitter.split_documents(docs)
    return (docs)

documents=chunk_data(docs=doc)
documents

# 1. Load a free HuggingFace embedding model
def get_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"   # free & small
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

embeddings = get_embeddings()


# 2. Example: embed all chunks (documents) and store in Pinecone
def store_embeddings_pinecone(documents, index_name):
    # initialize Pinecone
    pc = pinecone.Pinecone(api_key="pcsk_5CX4Px_31FNGf7oNNNcpn2qVuXJjQXAUjSvvtJAeD4cHNC2hBKVJXE84Qj1bEYJTiAfoyq")

    
    index = pc.Index(index_name)

    # create vectorstore
    vectorstore = PineconeVectorStore.from_documents(
        documents, embeddings, index_name=index_name
    )
    return vectorstore

# Call function
index_name = "chatbox"
vectorstore = store_embeddings_pinecone(documents, index_name)
print(" Embeddings created and stored successfully!")

# Load local LLM (Ollama)
def load_llm(model_name="llama3"):
    return Ollama(model=model_name)

llm = load_llm()

# Create retriever from Pinecone vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Example query
query = "What is the main purpose of the document?"
response = qa_chain.invoke({"query": query})

print("\n Final Answer:")
print(response["result"])

print("\n Sources used:")
for i, doc in enumerate(response["source_documents"], 1):
    print(f"\n--- Source {i} ---")
    print(doc.page_content[:500], "...")  # only print first chars


    


