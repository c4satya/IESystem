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

import os
import uuid
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone,ServerlessSpec
from Similarity_Serach import HSCodeRetriever 
from reportlab.lib import colors
from typing import Tuple, Optional


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
retriever = HSCodeRetriever(GEMINI_API_KEY+"",PINECONE_API_KEY+"","pdf-embeddings")






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
def search_hs_code(product_name: str,product_description:str):
    """
    Real web search HS classifier - scrapes DGFT/Trade portals
    Returns: (hs_code, explanation/confidence)
    """
    st.info("🔍 Searching HS codes from Vector Database ...")
            
    # Search queries (DGFT first, fallback global)
    search_query = f"HS code for {product_name} and having description {product_description}"
    # Clean query for search
    query = re.sub(r'[^\w\s]', ' ', search_query.lower()).strip()
    results = retriever.fetch_hs_code(product_name=product_name,product_description=product_description)
    st.info(f"HS_Code:{results["hs_code"]} and description: {results["product_desc"]}")
    return results


ans=search_hs_code("Makrana White Marble"," A world-renowned, pure white, high-calcite marble with long-lasting quality, ideal for premium flooring and luxury, classic architectural projects.")
print(ans)