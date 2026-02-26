import google.generativeai as genai
import os
from typing import List
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field
from pydantic import field_validator
from pydantic import ConfigDict  # ✅ NEW for v2

# Setup
load_dotenv()
# GEMINI_API_KEY = "AIzaSyDA_lygOBocCle26oFoDfb4bAHNaBmX9_I"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
logger = logging.getLogger(__name__)

class RouteStep(BaseModel):
    """Single step in export route sequence."""
    description: str = Field(..., description="Step description")
    mode: str = Field(..., description="transport mode: truck/sea/air/rail")
    duration_hours: int = Field(..., ge=1, description="Duration in hours")

class BestExportRoute(BaseModel):
    """Single BEST route for beginner exporter."""
    model_config = ConfigDict(  # ✅ FIXED: Pydantic v2 ConfigDict
        json_schema_extra={
            "examples": [
                {
                    "name": "Mumbai-JNPT-NYC Sea FCL",
                    "total_days": 28,
                    "total_cost_usd": 2500.0,
                    "total_cost_inr": 210000.0,
                    "sequence": [
                        {"description": "Mumbai Warehouse → JNPT Port", "mode": "truck", "duration_hours": 2},
                        {"description": "JNPT → New York Port (Maersk)", "mode": "sea", "duration_hours": 600},
                        {"description": "NYC Port → Buyer Warehouse", "mode": "truck", "duration_hours": 4}
                    ],
                    "why_best_for_beginners": "Cheapest bulk route, weekly sailings"
                }
            ]
        }
    )
    
    name: str = Field(..., description="Route name")
    total_days: int = Field(..., ge=1, le=60, description="Total transit days")
    total_cost_usd: float = Field(..., ge=100, le=50000, description="Total cost USD")
    total_cost_inr: float = Field(..., ge=8000, le=4200000, description="Total cost INR")
    sequence: List[RouteStep] = Field(..., min_length=2, max_length=5, description="Chain of route steps")  # ✅ FIXED
    why_best_for_beginners: str = Field(..., max_length=200, description="Why this route suits new exporters")
    
    @field_validator('total_cost_inr')
    @classmethod
    def validate_inr_consistency(cls, v: float, info):
        total_cost_usd = info.data.get('total_cost_usd')
        if total_cost_usd:
            expected = total_cost_usd * 84
            if abs(v - expected) > 100:
                raise ValueError(f"INR should be ~USD*84, got {v} vs expected {expected}")
        return v

class ExportRouteAnalyzer:
    def __init__(self, model_name: str = 'gemini-3-flash-preview'): 
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("Set GEMINI_API_KEY in .env")
        self.model = genai.GenerativeModel('gemini-3-flash-preview')

    def get_best_route(
        self,
        product_name: str,
        category: str,
        hs_code: str,
        quantity: int,
        export_country: str
    ) -> BestExportRoute:
        """
        Get SINGLE BEST route for beginner exporter.
        """
        prompt = self._build_single_best_prompt(product_name, category, hs_code, quantity, export_country)
        
        try:
            response = self.model.generate_content(prompt)
            json_str = self._extract_json(response.text)
            route_dict = json.loads(json_str)
            route = BestExportRoute(**route_dict)
            logger.info(f"✅ Best route validated: {route.name}")
            return route
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self._fallback_route(export_country, product_name)

    def _build_single_best_prompt(self, product_name: str, category: str, hs_code: str, 
                                 quantity: int, export_country: str) -> str:
        schema = BestExportRoute.model_json_schema()
        return f"""You are expert export consultant for NEW EXPORTERS from Mumbai, India.

INPUT:
Product: {product_name} | Category: {category} | HS: {hs_code} | Qty: {quantity} | To: {export_country}

TASK: Return ONLY the SINGLE BEST ROUTE for beginners.

OUTPUT EXACTLY ONE JSON object matching this schema (NO extra text):

{schema}

Realistic 2026 data ONLY:
- Sea FCL 20ft India-USA: $2200-3000, 25-35d
- Sea India-Europe: $2000-2800, 20-28d  
- Air Mumbai-Dubai: $8/kg, 1d
- Truck Mumbai-JNPT: ₹5-10k, 2h

JSON ONLY - will be parsed automatically."""

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract first valid JSON object."""
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1:
            raise ValueError("No JSON found")
        return text[start:end]

    def _fallback_route(self, country: str, product: str) -> BestExportRoute:
        """Smart fallback based on country/product."""
        base_days = 25
        base_cost = 2500
        
        if "USA" in country.upper() or "AMERICA" in country.upper():
            port = "New York"
        elif "EUROPE" in country.upper() or "GERMANY" in country.upper():
            port = "Rotterdam" 
            base_days = 22
            base_cost = 2400
        elif "DUBAI" in country.upper() or "UAE" in country.upper():
            port = "Jebel Ali"
            base_days = 8
            base_cost = 1200
        else:
            port = "Main Port"
        
        return BestExportRoute(
            name=f"Best {product} Route to {country}",
            total_days=base_days,
            total_cost_usd=base_cost,
            total_cost_inr=base_cost * 84,
            sequence=[
                RouteStep(description=f"Mumbai Warehouse → JNPT Port", mode="truck", duration_hours=2),
                RouteStep(description=f"JNPT → {port} ({'Maersk' if base_days > 15 else 'Direct'})", 
                         mode="sea" if base_days > 10 else "air", duration_hours=base_days*24),
                RouteStep(description=f"{port} → Buyer Warehouse", mode="truck", duration_hours=4)
            ],
            why_best_for_beginners=f"Optimized for {product} beginners to {country} - reliable major carriers"
        )

# Usage
def main():
    analyzer = ExportRouteAnalyzer(model_name='gemini-1.5-flash')  # ✅ Available model
    
    route = analyzer.get_best_route(
        product_name="Cotton T-shirts",
        category="Apparel", 
        hs_code="61091000",
        quantity=1000,
        export_country="United States"
    )
    
    print("✅ BEST ROUTE FOR BEGINNERS:")
    print(route.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
