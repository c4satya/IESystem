import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import json
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Mock ML
import hashlib 
import requests
from bs4 import BeautifulSoup
import re
import time
from typing import Tuple, Optional
import os
import uuid
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone,ServerlessSpec
from Similarity_Serach import HSCodeRetriever 


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
retriever = HSCodeRetriever(GEMINI_API_KEY+"",PINECONE_API_KEY+"","pdf-embeddings")

results = retriever.fetch_hs_code(
    product_name="Basmati Rice",
    product_description="Premium long grain basmati rice for export"
)

for r in results:
    print(r)


def generate_pdf(product_name, product_desc, country, qty, unit_type, hs, compliance, costs, time_days, risk, unit_price=10):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()  # ✅ Available styles only
    
    story = []
    
    # Header
    story.append(Paragraph("GlobalTradeAI - Complete Export Analysis Report", styles['Title']))
    story.append(Spacer(1, 20))
    
    # ✅ FIXED: User Inputs Section - Use 'Heading2' and 'Normal'
    story.append(Paragraph("1. User Inputs", styles['Heading2']))
    story.append(Paragraph(f"Product Name: <b>{product_name}</b>", styles['Normal']))
    story.append(Paragraph(f"Description: {product_desc}", styles['Normal']))
    story.append(Paragraph(f"Destination: {country}", styles['Normal']))
    # story.append(Paragraph(f"Quantity: {qty} {units_options[unit_type]}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Rest unchanged...
    story.append(Paragraph("2. AI Analysis", styles['Heading2']))
    story.append(Paragraph(f"HS Code: <b>{hs}</b>", styles['Normal']))
    story.append(Paragraph(f"Compliance: {compliance['status']}", styles['Normal']))
    story.append(Paragraph(f"Risk Score: <b>{risk}/100</b>", styles['Normal']))
    # ... continue with costs table
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# HS-CODE CLASSIFIER



@st.cache_data(ttl=3600)  # Cache 1hr
def search_hs_code(product_query: str) -> Tuple[Optional[str], str]:
    """
    Real web search HS classifier - scrapes DGFT/Trade portals
    Returns: (hs_code, explanation/confidence)
    """
    st.info("🔍 Searching HS codes from DGFT/Trade.gov.in...")
    
    # Clean query for search
    query = re.sub(r'[^\w\s]', ' ', product_query.lower()).strip()
    
    # Search queries (DGFT first, fallback global)
    search_queries = [
        f"HS code {query} India DGFT",
        f"HS code for {query}",
        f"{query} export HS code"
    ]
    return None
def is_valid_hs(hs_code: str) -> bool:
    if not hs_code or hs_code == "N/A":
        return False
    return bool(re.match(r'^\d{4}(?:\.\d{2})?\d{0,4}$', str(hs_code))) 



# PDF EMBEDDING IN VECTOR DB


# # Load env variables
# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "pdf-embeddings"

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Create index if not exists
# if INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=384,  # all-MiniLM-L6-v2 embedding size
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# index = pc.Index(INDEX_NAME)

# # 1️⃣ Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text() + "\n"
#     return text

# # 2️⃣ Chunk text
# def chunk_text(text, chunk_size=500, overlap=50):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - overlap
#     return chunks

# # 3️⃣ Create embeddings
# def embed_chunks(chunks):
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(chunks)
#     return embeddings

# # 4️⃣ Upload to Pinecone
# def upload_to_pinecone(chunks, embeddings):
#     vectors = []
#     for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#         vectors.append({
#             "id": str(uuid.uuid4()),
#             "values": embedding.tolist(),
#             "metadata": {"text": chunk}
#         })

#     index.upsert(vectors=vectors)

# # 🚀 Full Pipeline
# if __name__ == "__main__":
#     pdf_path = "sample.pdf"

#     print("Extracting text...")
#     text = extract_text_from_pdf(pdf_path)

#     print("Chunking...")
#     chunks = chunk_text(text)

#     print("Creating embeddings...")
#     embeddings = embed_chunks(chunks)

#     print("Uploading to Pinecone...")
#     upload_to_pinecone(chunks, embeddings)

#     print("✅ PDF successfully embedded into Pinecone.")