from fastapi import FastAPI, Query
from pydantic import BaseModel
import os
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# Constants for Chroma settings
CHROMA_SETTINGS = {"index_type": "flat", "metric_type": "l2"}
persist_directory = "db"
similarity_threshold = 1.5

# Initialize Chroma client
chroma_client = chromadb.Client()

# Create or get a collection
collection = chroma_client.get_or_create_collection(name="my_collection")

# Load PDFs and initialize embeddings
pdf_files = ["WhatYouNeedToKnowAboutWOMENSHEALTH.pdf", "Clinical Manual of Women's Mental Health.pdf"]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

for pdf_file in pdf_files:
    if not os.path.isfile(pdf_file):
        raise ValueError(f"PDF file '{pdf_file}' not found.")

    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    texts = [doc.page_content for doc in documents]

    # Split text into chunks
    docs = text_splitter.split_documents(documents)
    content_for_embedding = [doc.page_content for doc in docs]
    embeddings = embeddings_model.embed_documents(content_for_embedding)
    metadatas = [{"source": pdf_file, "page_number": i + 1, "content": content} for i, content in enumerate(content_for_embedding)]
    ids = [f"{pdf_file}_{i}" for i in range(len(docs))]
    
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

# Load model and tokenizer
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

class QueryModel(BaseModel):
    query: str

def truncate_context(context, max_words=30):
    words = context.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return context

def chat(query: str):
    results = collection.query(query_texts=[query], n_results=2)
    documents = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]
    metadata_list = results.get('metadatas', [[]])[0]

    context = " ".join(documents)
    truncated_context = truncate_context(context, max_words=30)
    sources = [meta.get('source', 'Unknown') if meta else 'Unknown' for meta in metadata_list]
    source_info = f"Source: {', '.join(sources)}"

    if distances and distances[0] < similarity_threshold:
        prompt_with_context = f"Context Page: {truncated_context}\n{source_info}\n\nQuestion: {query}\nAnswer:"
        inputs = tokenizer(prompt_with_context, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=70, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f"Answer with document context: {answer}\n{source_info}"
    return "The answer to your query is not found in the provided documents."

@app.post("/predict")
async def predict(query: QueryModel):
    response = chat(query.query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
