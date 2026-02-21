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
    story.append(Paragraph(f"Quantity: {qty} {units_options[unit_type]}", styles['Normal']))
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
