from dotenv import load_dotenv
from pypdf import PdfReader
import uuid
import os
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import time

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "pdf-embeddings-768"

# genai.configure(api_key=GEMINI_API_KEY)

# DELETE OLD INDEX FIRST (CRITICAL)
pc = Pinecone(api_key=PINECONE_API_KEY)
# if INDEX_NAME in pc.list_indexes().names():
#     print("🗑️  Deleting old 768-dim index...")
#     pc.delete_index(INDEX_NAME)
#     time.sleep(5)

# # Create FRESH 768-dim index
# print("🆕 Creating fresh 768-dim index...")
# pc.create_index(
#     name=INDEX_NAME,
#     dimension=768,  # ✅ FIXED
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# )
index = pc.Index(INDEX_NAME)
# time.sleep(3)

# ✅ RIGHT MODEL FOR 768-DIM
model = SentenceTransformer("all-mpnet-base-v2")  # 768-dim, NOT MiniLM

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=400, overlap=20):  # Smaller chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed_chunks(chunks):
    print(f"🔄 Embedding {len(chunks)} chunks with 768-dim model...")
    embeddings = model.encode(chunks, batch_size=16, show_progress_bar=True)
    print(f"✅ Created {len(embeddings)} embeddings, shape: {embeddings.shape}")
    return embeddings

def upload_to_pinecone_batched(chunks, embeddings):
    """✅ BATCHED - Fixes 4MB limit + dimension error"""
    batch_size = 25  # Smaller batches
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch_vectors = []
        end = min(i + batch_size, total)
        
        for j in range(i, end):
            vector = {
                "id": f"doc1_chunk_{i+j}",
                "values": embeddings[j].tolist(),
                "metadata": {"text": chunks[j][:800]}  # Truncate metadata
            }
            batch_vectors.append(vector)
        
        print(f"📤 Batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch_vectors)} vectors)")
        index.upsert(vectors=batch_vectors)
        time.sleep(0.2)  # Rate limit
    
    print("✅ ALL vectors uploaded!")

# Search functions
def embed_query(query):
    return model.encode([query])[0].tolist()

def search_similar_chunks(query, top_k=3):
    query_emb = embed_query(query)
    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    for match in results["matches"]:
        print(match)

    return [match['metadata']['text'] for match in results['matches']]

# ✅ FIXED Gemini - Multiple fallbacks
# def generate_response_with_gemini(query, context_chunks):
#     context = "\n\n".join(context_chunks)
#     prompt = f"""From this trade statistics PDF, answer:

# CONTEXT:
# {context}

# Q: {query}
# A:"""
    
#     # ✅ WORKING MODELS (2026 stable)
#     models_to_try = ['gemini-pro-latest', 'gemini-2.5-pro']
    
#     for model_name in models_to_try:
#         try:
#             print(f"🤖 Trying {model_name}...")
#             gemini_model = genai.GenerativeModel(model_name)
#             response = gemini_model.generate_content(prompt)
#             print(f"✅ {model_name} SUCCESS!")
#             return response.text
#         except Exception as e:
#             print(f"❌ {model_name}: {str(e)[:100]}")
#             continue
    
#     return "No working Gemini model available. Check API key/project."

# 🚀 EXECUTE
# if __name__ == "__main__":
    # pdf_path = r"C:\Users\USER\Downloads\HS_CODE_REF-pages\HS_CODE_REF-pages-4.pdf"
    
    # print("📋 Available Gemini models:")
    # for m in genai.list_models():
    #     if 'generateContent' in m.supported_generation_methods:
    #         print(f"  ✅ {m.name}")
    
    # print("📄 1. Extracting...")
    # text = extract_text_from_pdf(pdf_path)
    
    # print("✂️  2. Chunking...")
    # chunks = chunk_text(text)
    # print(f"📦 {len(chunks)} chunks created")
    
    # print("🧠 3. Embedding (768-dim)...")
    # embeddings = embed_chunks(chunks)
    
    # print("\n☁️  4. Uploading (batched)...")
    # upload_to_pinecone_batched(chunks, embeddings)
    
    # print("\n🎉 SUCCESS! Testing search...")
    
    
    
    
    
   
