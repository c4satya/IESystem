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
from sklearn.ensemble import RandomForestRegressor 
import hashlib 
from datetime import datetime
import os
from appFunctions import * 
import re

from export_complinaces import compliance_pipeline
from route_analysis import ExportRouteAnalyzer
 # Mock HS classifier

# Mock data simulating integrations (DGFT/IEC, ICEGATE, RBI, WTO, OFAC, PCS, ECGC)
if 'results' not in st.session_state:
    st.session_state.results = {}

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
MOCK_HS_DATA = {
    'textiles': '5208.12', 'spices': '0904.11', 'electronics': '8517.62', 'software': '8523.49'
}
MOCK_LICENSES = {'5208.12': ['IEC', 'Textile Export License'], '0904.11': ['IEC', 'Phytosanitary', 'FSSAI']}
MOCK_DUTIES = {'USA': 0.05, 'EU': 0.08, 'China': 0.10, 'Singapore': 0.00}
MOCK_INCENTIVES = ['RoDTEP 4%', 'NIRYAT PROTSAHAN 2.75%', 'ECGC Insurance Eligible']
MOCK_PORT_CONGESTION = {'Mumbai': 2, 'Chennai': 3}  # Days delay


# Mock Knowledge Graph query (Neo4j sim: Product → HS → Country → Regs)
category=""
def mock_kg_query(hs, country):
    compliance_result=compliance_pipeline(hs_code,country,product_name,product_desc)
    licenses = MOCK_LICENSES.get(hs, ['IEC'])
    duty = MOCK_DUTIES.get(country, 0.06)
    return {'licenses': licenses, 'duties': duty, 'restricted': 'china' in country.lower()}






@st.cache_data
def predict_time(hs, country, qty, port='Mumbai'):
    

# Risk Engine (Random Forest scoring)
def risk_score(product: str, country: str, hs: str = None) -> int:
    risk_S=
    return    
# Compliance Validator (Mock DGFT/ICEGATE/RBI/OFAC)
def compliance_check(hs, country, iec='mock123'):
    kg = ""
    issues = []
    if 'china' in country.lower():
        issues.append("Sanctions check: High risk (OFAC mock)")
    if not hs.startswith(('52', '09', '85')):
        issues.append("HS invalid per WTO")
    return {
        'status': '🟢 Compliant' if not issues else '🔴 Issues Found',
        'iec_valid': True,  # Mock DGFT API
        'gst_valid': True,  # Mock GSTN
        'issues': issues,
        'licenses': kg['licenses']
    }

# Cost Estimator (Freight + Duties + Insurance)
def estimate_costs(qty, country, unit_type=None, unit_price=10):
    """
    Complete unit-aware cost calculator (4 args supported)
    Compatible: estimate_costs(qty, country) OR estimate_costs(qty, country, unit_type, unit_price)
    """
    # Mock duties (DGFT/WTO - inline, no external deps)
    MOCK_DUTIES = {'USA': 0.05, 'EU': 0.08, 'China': 0.10, 'Singapore': 0.00, 'UK': 0.06, 'UAE': 0.05}
    duty_rate = MOCK_DUTIES.get(country, 0.06)
    
    # Unit volume factors for freight (Maersk/PCS)
    volume_factors = {'kg': 1.2, 'units': 1.0, 'L': 1.1, 'm': 0.8, 'box': 1.5}
    vol_factor = volume_factors.get((unit_type or 'units').split('/')[0], 1.0)
    
    # Calculations
    product_cost = unit_price * qty
    freight = (2 + qty * vol_factor / 1000) * 1000  # Base + volume
    duties = duty_rate * product_cost
    insurance = product_cost * 0.01  # ECGC standard
    
    total_landed = product_cost + freight + duties + insurance
    
    return {
        'product_cost': round(product_cost, 0),
        'freight': round(freight, 0),
        'duties': round(duties, 0),
        'insurance': round(insurance, 0),
        'total_landed': round(total_landed, 0),
        'duty_rate': f"{duty_rate*100:.1f}%",
        'unit_type': unit_type or 'units'
    }

# PDF Report Generator (Summary of all analysis)
def generate_pdf(product_name, product_desc, country, qty, unit_type, hs, compliance, costs, time_days, risk, unit_price=10):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()  # ✅ Available styles only
    
    story = []

    story.append(Paragraph("GlobalTradeAI - Complete Export Analysis Report", styles['Title']))
    story.append(Spacer(1, 20))
    
     # ✅ FIXED: User Inputs Section - Use 'Heading2' and 'Normal'
    story.append(Paragraph("1. User Inputs", styles['Heading2']))
    story.append(Paragraph(f"Product Name: <b>{product_name}</b>", styles['Normal']))
    story.append(Paragraph(f"Description: {product_desc}", styles['Normal']))
    story.append(Paragraph(f"Destination: {country}", styles['Normal']))
    story.append(Paragraph(f"Quantity: {qty} {units_options[unit_type]}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # AI Outputs
    story.append(Paragraph("2. AI Analysis", styles['Heading2']))
    story.append(Paragraph(f"• HS Code (AI Classified): {hs}", styles['Normal']))
    story.append(Paragraph(f"• Compliance Status: {compliance['status']}", styles['Normal']))
    if compliance['issues']:
        story.append(Paragraph("Issues: " + "; ".join(compliance['issues']), styles['Normal']))
    story.append(Paragraph(f"• Licenses Required: {', '.join(compliance['licenses'])}", styles['Normal']))
    story.append(Paragraph(f"• Risk Score: {risk}/100", styles['Normal']))
    story.append(Paragraph(f"• Predicted Time: {time_days} days (XGBoost model)", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Costs Table
    story.append(Paragraph("3. Cost Breakdown (₹)", styles['Heading2']))
    cost_data = [
        ['Component', 'Amount'],
        ['Product Cost', f"{costs['product_cost']:,.0f}"],
        ['Freight (PCS/Maersk)', f"{costs['freight']:,.0f}"],
        ['Duties (WTO/DGFT)', f"{costs['duties']:,.0f}"],
        ['Insurance (ECGC)', f"{costs['insurance']:,.0f}"],
        ['Total Landed Cost', f"{costs['total_landed']:,.0f}"]
    ]
    table = Table(cost_data, colWidths=[2.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(table)
    
    # Recommendations
    story.append(Spacer(1, 20))
    story.append(Paragraph("4. Recommendations", styles['Heading2']))
    recs = [
        "Incoterm: FOB (Low risk for beginners)",
        "Route: Mumbai Port (PCS integrated)",
        "Payment: LC (RBI compliant)",
        "Incentives: RoDTEP, NIRYAT (DGFT eligible)"
    ]
    for rec in recs:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Streamlit Frontend (Full Dashboard)
st.set_page_config(page_title="GlobalTradeAI", page_icon="🌍", layout="wide")
st.title("🌍 GlobalTradeAI – Intelligent Export Orchestration Platform")
st.markdown("**AI Export Consultant** for new exporters: HS Classification, Compliance, Costs, Risks, Predictions & PDF Reports.")

# Sidebar: Beginner Mode / Tech Info
with st.sidebar:
    st.header("🧭 Beginner Mode")
    st.info("1. Describe product\n2. Select country\n3. Enter qty\n4. Get full plan!")
    st.header("🔗 Integrations (Mock APIs)")
    st.markdown("- 🇮🇳 DGFT (IEC/Licenses)\n- ICEGATE (Clearance)\n- PCS (Ports)\n- RBI (Forex)\n- WTO/UN Comtrade\n- OFAC/ECGC")
    st.header("🧠 AI Stack")
    st.markdown("- HS: Transformer-mock\n- KG: Neo4j-sim\n- Predict: XGBoost/RF\n- Copilot: LLM-ready")
    st.markdown("---")
    st.caption("MVP | Deploy: `streamlit run app.py`")
def validate_inputs(product_name, product_desc, country, qty, unit_type):
    if not product_name or product_name.strip() == "":
        return False, "❌ **Product Name** is mandatory (invoice reference)."
    if not product_desc or product_desc.strip() == "":
        return False, "❌ **Product Description** is **mandatory** for AI HS classification & compliance checks."
    if not country:
        return False, "❌ Destination country required."
    if qty < 1:
        return False, f"❌ Quantity ≥1 {units_options.get(unit_type, 'unit')}."
    return True, "✅ Complete - analyzing..."

# Main Inputs
units_options = {
    'Pieces/Units': 'units',
    'Kilograms': 'kg', 
    'Liters': 'L',
    'Meters': 'm',
    'Boxes': 'box'
}

col1, col2 = st.columns([2, 1])
col1a, col1b, col3 = st.columns([2, 3, 1])
with col1a:
    product_name = st.text_input(
        "Product Name", 
        placeholder="e.g., Cotton T-Shirt, Turmeric Powder", 
        max_chars=50,
        help="Short commercial name (for invoice/PDF)"
    )
with col1b:
    product_desc = st.text_input(
        "Product Description", 
        placeholder="e.g., 100% cotton crew neck T-shirt size M, or Organic turmeric powder 500g packs", 
        help="**Detailed specs** for AI HS classification & compliance"
    )

with col3:
    
    country = st.selectbox("Destination Country", ["USA", "EU", "China", "Singapore", "UK"])


col1, col2,col3 = st.columns([1, 1, 1])
with col1:
    qty = st.number_input("Quantity (units)", min_value=1, value=1000, help="Affects costs/time")
with col2:    
    unit_type = st.selectbox("Unit Type", list(units_options.keys()), index=0)
with col3:    
    unit_price = st.number_input("Unit Price (₹)", min_value=0.1, value=10.0)




if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
    # Validation...
    
    if not st.session_state.results:
        st.session_state.results = {}    
    
    st.spinner("🔍 HS → Compliance → Route → Costs → Time →...")
    hs_result = search_hs_code(product_name,product_desc)
    hs_code= hs_result["hs_code"]
    if hs_result:
        st.header("HS CODE FOUND\n"+hs_result["hs_code"]+" : "+hs_result["product_desc"])

    
    
    # Safe calls - pass None-safe values
    compliance = compliance_pipeline(hs_code,country,product_name,product_desc)
    
    category=compliance["category"]
    
    analyzerR = ExportRouteAnalyzer()
    route = analyzerR.get_best_route(product_name,category,hs_code,quantity,export_country)
    cost=estimate_costs(qty, country, unit_type, unit_price)
    
    # time_days = predict_time(hs_code, country, qty)
    # risk = risk_score(full_product, country, hs_code)  
    
    # Store results
    st.session_state.results = {
        'hs': hs_code,
        'product_desc': hs_result["product_desc"],
        'compliance': compliance,
        'costs': 0,
        'time_days': 0,
        'risk': "20%"
    }
        
        # Cache results
    st.session_state.results = {
            'hs': hs_code, 'compliance': compliance, 'costs': costs,
            'time_days': time_days, 'risk': risk
        }

    # Results Dashboard
    st.header("📊 Decision Intelligence Dashboard")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Risk Score", f"{st.session_state.results['risk']}/100", "🟢 Safe" if st.session_state.results['risk'] < 40 else "🔴 High Risk")
    col_b.metric("Clearance Time", f"{st.session_state.results['time_days']} days")
    col_c.metric("Total Cost", f"₹{st.session_state.results['costs']['total_landed']:,.0f}")
    col_d.metric("Compliance", st.session_state.results['compliance']['status'])

    st.header("✅ Compliance Checker")
    st.json(st.session_state.results['compliance'])
    
    st.header("💰 Cost Estimator (₹)")
    # ✅ FIXED: Pure numeric DataFrame - NO styling conflicts
    numeric_costs = {
        'Component': ['Product Cost', 'Freight (Maersk/PCS)', 'Duties (DGFT/WTO)', 
                    'Insurance (ECGC)', 'Total Landed Cost'],
        'Amount (₹)': [
            costs['product_cost'], costs['freight'], costs['duties'], 
            costs['insurance'], costs['total_landed']
        ]
    }
    cost_df = pd.DataFrame(numeric_costs)

    # ✅ Number formatting ONLY - no background_gradient
    st.table(cost_df.style.format({'Amount (₹)': '{:,.0f}'}))
    # st.header("💰 Cost Estimator")
    # cost_df = pd.DataFrame(list(st.session_state.results['costs'].items()), columns=['Component', '₹ Amount'])
    # st.table(cost_df.style.format({'₹ Amount': '{:,.0f}'}))
    
    
    st.header("🧭 Recommendations")
    st.success("• Best Incoterm: FOB\n• Route: Mumbai PCS → Optimal\n• Payment: LC\n• Financing: ECGC Packing Credit")

    # PDF Download
    st.header("📥 Download Report")
    pdf_data = generate_pdf(product_name, product_desc, country, qty, unit_type, 
                       st.session_state.results['hs'], 
                       st.session_state.results['compliance'], 
                       st.session_state.results['costs'], 
                       st.session_state.results['time_days'], 
                       st.session_state.results['risk'])
    
    safe_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '-', '_')).rstrip()[:20]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"GlobalTradeAI_{safe_name}_{country}_{timestamp}.pdf"
    
    st.download_button(
    label="💾 Download Complete Analysis PDF",
    data=pdf_data,
    file_name=filename,
    mime="application/pdf"
)
