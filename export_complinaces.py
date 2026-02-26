import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel
from typing import List

import re


def clean_llm_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)

class ComplianceResponse(BaseModel):
    hs_code: str
    destination_country: str
    product_category: str
    india_export_compliances: List[str]
    destination_import_compliances: List[str]

def clean_llm_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


def normalize_hs(hs_code: str) -> str:
    hs = hs_code.strip().replace(".", "")
    if len(hs) != 8 or not hs.isdigit():
        raise ValueError("Invalid Indian HS Code")
    return hs

def rule_engine(hs_code: str) -> list:
    prefix = hs_code[:2]
    # ---------------- RULE ENGINE DATABASE ---------------- #

    COMPLIANCE_RULES = {
    "01": ["Animal Quarantine Certificate"],
    "02": ["Animal Quarantine Clearance"],
    "03": ["Marine Products Export Development Authority (MPEDA) Registration"],
    "07": ["Plant Quarantine Certificate"],
    "10": ["Phytosanitary Certificate"],
    "15": ["FSSAI Export License"],
    "16": ["FSSAI Export License"],
    "17": ["FSSAI Export License"],
    "18": ["FSSAI Export License"],
    "19": ["FSSAI Export License"],
    "20": ["FSSAI Export License"],
    "21": ["FSSAI Export License"],
    "22": ["FSSAI Export License"],
    "30": ["CDSCO Export NOC"],
    "33": ["Pharmaceutical Export NOC"],
    "38": ["Chemical Export Declaration"],
    "50": ["Textile Committee Certification"],
    "61": ["Textile Committee Certification"],
    "62": ["Textile Committee Certification"],
    "63": ["Textile Committee Certification"],
    "71": ["Kimberley Process Certificate"],
    "84": ["BIS Certification (if notified under QCO)"],
    "85": ["WPC ETA Approval", "BIS CRS"],
    "90": ["CDSCO Medical Device Export Approval"],
}
    return COMPLIANCE_RULES.get(prefix, [])

def merge_compliances(rule_compliances: list, llm_compliances: list) -> list:
    """
    Merge rule-engine and LLM-generated compliances into one clean list.
    Output format: List[str] (one-liner compliance strings)
    """

    final = set(llm_compliances)

    for rule in rule_compliances:
        final.add(f"{rule} – Mandatory statutory export compliance")

    return sorted(final)

def compliance_pipeline(hs_code: str, destination_country: str,product_name:str,product_desc:str) -> dict:
    hs_code = normalize_hs(hs_code)

    rule_results = rule_engine(hs_code)

    llm_result = get_country_wise_compliances(hs_code, destination_country,product_name,product_desc)

    if "error" in llm_result:
        return llm_result

    merged_india = merge_compliances(
    rule_results,
    llm_result.get("india_export_compliances", []))
    llm_result["india_export_compliances"] = merged_india

    return llm_result

COUNTRY_COMPLIANCE_PROMPT = """
        You are a global trade compliance expert.

        Given:
        - Indian 8-digit HS code
        - Product name and description
        - Destination country

        Return STRICT JSON with ONLY ONE-LINER compliance outputs.

        Rules:
        - Each compliance must be a single-line string.
        - Format: "Compliance Name - Short description"
        - No markdown
        - No explanations
        - No bullets
        - No extra text

        Return JSON exactly in this format:

        {{
        "hs_code": "{hs_code}",
        "destination_country": "{country}",
        "product_category": "",
        "india_export_compliances": [
            "Compliance Name – description"
        ],
        "destination_import_compliances": [
            "Compliance Name – description"
        ]
        }}

        HS Code: {hs_code}
        Product Name: {product_name}
        Product Description: {product_desc}
        Destination Country: {country}
        """


def get_country_wise_compliances(hs_code, country, product_name, product_desc):
    
    load_dotenv
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-3-flash-preview")

    prompt = COUNTRY_COMPLIANCE_PROMPT.format(
        hs_code=hs_code,
        country=country,
        product_name=product_name,
        product_desc=product_desc
    )

    response = model.generate_content(prompt)

    cleaned = clean_llm_json(response.text)

    return ComplianceResponse(**cleaned).model_dump(mode="json")


# function call


# if st.button("Check Full Export Compliance"):
#     with st.spinner("Analyzing Indian + Import Country Regulations..."):
result = compliance_pipeline("68022110","USA","MARBLE BLOCKS/TILES,POLISHED","Polished Makrana white marble blocks and tiles for premium flooring")

print(result["hs_code"])
print(result["destination_country"])
print(result["product_category"])
print("india_export_compliances:")
for i in result["india_export_compliances"]:
    print(i)
print("destination_import_compliances")
for i in result["destination_import_compliances"]:
    print(i)
        