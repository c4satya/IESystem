import os
import json
import logging
from typing import List
from dotenv import load_dotenv

import google.generativeai as genai
from pydantic import BaseModel, Field, ConfigDict

# -------------------- Setup --------------------

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_route_ai")

# -------------------- Data Models --------------------

class RouteStep(BaseModel):
    description: str = Field(..., description="Route segment")
    mode: str = Field(..., description="truck / sea / air / rail")
    duration_hours: int = Field(..., ge=1, le=2000)


class BestExportRoute(BaseModel):
    """
    Best beginner export route (INR only).
    """
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=10, description="Full route path with → separators")
    total_days: int = Field(..., ge=1, le=60)
    total_cost_inr: float = Field(..., ge=8000, le=4500000, description="ALL COSTS IN INR")
    sequence: List[RouteStep] = Field(..., min_length=2, max_length=5)
    why_best_for_beginners: str = Field(..., min_length=10, max_length=200)

# -------------------- Analyzer --------------------

class ExportRouteAnalyzer:
    """
    AI-powered export route planner for Indian exporters.
    Outputs ONLY INR currency.
    """

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("❌ GEMINI_API_KEY missing in .env")

        self.model = genai.GenerativeModel(model_name)

    def get_best_route(
        self,
        product_name: str,
        category: str,
        hs_code: str,
        quantity: int,
        export_country: str
    ) -> BestExportRoute:

        prompt = self._build_prompt(
            product_name, category, hs_code, quantity, export_country
        )

        try:
            response = self.model.generate_content(prompt)
            json_str = self._extract_json(response.text)
            route_dict = json.loads(json_str)

            route = BestExportRoute(**route_dict)
            logger.info(f"✅ Valid route: {route.name}")
            return route

        except Exception as e:
            logger.error(f"❌ AI route failed: {e}")
            return self._fallback_route(export_country, product_name)

    # -------------------- Prompt --------------------

    def _build_prompt(
        self,
        product_name: str,
        category: str,
        hs_code: str,
        quantity: int,
        export_country: str
    ) -> str:

        schema = BestExportRoute.model_json_schema()

        return f"""
You are a SENIOR EXPORT LOGISTICS CONSULTANT in Mumbai, India.

INPUT:
Product: {product_name}
Category: {category}
HS Code: {hs_code}
Quantity: {quantity}
Destination Country: {export_country}

TASK:
Return ONLY the SINGLE BEST EXPORT ROUTE for a FIRST-TIME EXPORTER.

STRICT RULES:
- ALL COSTS MUST BE IN INDIAN RUPEES (INR).
- DO NOT mention USD anywhere.
- Provide REALISTIC 2026 logistics pricing.
- name must show full path using → arrows.
- Output ONLY valid JSON. No explanations.

REFERENCE COSTS (2026 realistic):
- Mumbai → JNPT trucking: ₹5,000 – ₹12,000
- India → USA sea (20ft FCL): ₹1.9L – ₹2.7L, 25–35 days
- India → Europe sea: ₹1.7L – ₹2.5L, 20–28 days
- India → Dubai sea: ₹65k – ₹1.1L, 7–10 days
- India → Dubai air: ₹550 – ₹700 per kg

JSON SCHEMA:
{schema}

Return JSON only.
"""

    # -------------------- JSON Handling --------------------

    @staticmethod
    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("❌ No JSON detected in LLM response")

        return text[start:end]

    # -------------------- Fallback --------------------

    def _fallback_route(self, country: str, product: str) -> BestExportRoute:
        logger.warning("⚠ Using fallback logistics model")

        base_days = 28
        base_cost = 225000

        if "USA" in country.upper():
            port = "New York"
            base_cost = 240000
        elif "GERMANY" in country.upper() or "EUROPE" in country.upper():
            port = "Rotterdam"
            base_cost = 220000
            base_days = 24
        elif "UAE" in country.upper() or "DUBAI" in country.upper():
            port = "Jebel Ali"
            base_cost = 95000
            base_days = 9
        else:
            port = "Main Port"

        return BestExportRoute(
            name=f"Mumbai Warehouse → JNPT Port → {port} → Buyer Warehouse",
            total_days=base_days,
            total_cost_inr=base_cost,
            sequence=[
                RouteStep(
                    description="Mumbai Warehouse → JNPT Port",
                    mode="truck",
                    duration_hours=2
                ),
                RouteStep(
                    description=f"JNPT → {port} (Container Shipping)",
                    mode="sea",
                    duration_hours=base_days * 24
                ),
                RouteStep(
                    description=f"{port} → Buyer Warehouse",
                    mode="truck",
                    duration_hours=4
                )
            ],
            why_best_for_beginners="Lowest risk, reliable carriers, predictable transit, simple customs clearance"
        )

# -------------------- Example Usage --------------------

def main():
    analyzerR = ExportRouteAnalyzer()

    route = analyzerR.get_best_route(
        product_name="Cotton T-Shirts",
        category="Apparel",
        hs_code="61091000",
        quantity=1000,
        export_country="United States"
    )

    print("\n✅ BEST EXPORT ROUTE (INR ONLY):")
    print(route.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
