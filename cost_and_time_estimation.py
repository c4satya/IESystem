from pydantic import BaseModel, Field, ConfigDict
from typing import List


class CostBreakdown(BaseModel):
    freight_inr: float
    port_charges_inr: float
    documentation_inr: float
    compliance_inr: float
    insurance_inr: float
    contingency_inr: float


class TimeBreakdown(BaseModel):
    documentation_days: int
    compliance_days: int
    transit_days: int
    customs_days: int


class RiskFactor(BaseModel):
    factor: str
    risk_score: float


class ExportEstimationLLMResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_estimated_cost_inr: float = Field(..., ge=5000, le=1e7)
    total_estimated_time_days: int = Field(..., ge=1, le=120)
    risk_percentage: float = Field(..., ge=0, le=100)

    cost_breakdown: CostBreakdown
    time_breakdown: TimeBreakdown
    risk_factors: List[RiskFactor]

import json
import os
import logging
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

# Import your existing BestExportRoute model
# from export_route_module import BestExportRoute

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logger = logging.getLogger("ai_estimation_engine")


class AIExportEstimationEngine:
    """
    Gemini-powered AI estimation & risk prediction engine.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)

    def estimate(
        self,
        product_name: str,
        product_description: str,
        hs_code: str,
        product_type: str,
        quantity: int,
        export_country: str,
        compliances: List[str],
        route
    ) -> ExportEstimationLLMResult:

        prompt = self._build_prompt(
            product_name,
            product_description,
            hs_code,
            product_type,
            quantity,
            export_country,
            compliances,
            route
        )

        try:
            response = self.model.generate_content(prompt)
            json_str = self._extract_json(response.text)
            data = json.loads(json_str)

            result = ExportEstimationLLMResult(**data)
            logger.info("✅ AI estimation validated successfully")
            return result

        except Exception as e:
            logger.error(f"❌ AI estimation failed: {e}")
            raise

    # ---------------- PROMPT ----------------

    def _build_prompt(
        self,
        product_name: str,
        product_description: str,
        hs_code: str,
        product_type: str,
        quantity: int,
        export_country: str,
        compliances: List[str],
        route
    ) -> str:

        schema = ExportEstimationLLMResult.model_json_schema()

        return f"""
You are a SENIOR EXPORT COST & RISK ANALYST based in Mumbai, India.

Your job is to predict TOTAL EXPORT COST, TIME, and RISK.

INPUT DATA:
Product Name: {product_name}
Description: {product_description}
HS Code: {hs_code}
Product Type: {product_type}
Quantity: {quantity}
Destination Country: {export_country}

Export Route:
{route.model_dump_json(indent=2)}

Compliance List:
{compliances}

RULES:
- All costs must be strictly in INR.
- Provide realistic 2026 logistics & compliance costs.
- Predict delays and risks intelligently.
- Risk % must reflect customs, product sensitivity, compliance complexity & country strictness.
- Output only valid JSON matching schema.
- NO explanations, NO markdown.

OUTPUT JSON SCHEMA:
{schema}
"""

    # ---------------- JSON Extraction ----------------

    @staticmethod
    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("No valid JSON returned by LLM")

        return text[start:end]


estimator = AIExportEstimationEngine()

result = estimator.estimate(
    product_name="Cotton T-Shirts",
    product_description="100% cotton knitted casual wear",
    hs_code="61091000",
    product_type="textile",
    quantity=1000,
    export_country="United States",
    compliances=["IEC", "GST", "Commercial Invoice", "Packing List"],
    route=null
)

print(result.model_dump_json(indent=2))