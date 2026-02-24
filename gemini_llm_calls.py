import google.generativeai as genai
from dotenv import load_dotenv
import os
from Embeding import *
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def get_best_hs_code(user_query: str, vector_results: list) -> str:
    """
    Returns the most accurate 8-digit HS code based on vector DB semantic search results.If hs code correct for given name and description provide hs code based on your knowledge

    :param user_query: Original product search query from user that contain product name and product description
    :param vector_results: List of dicts -> [{"hs_code": "xxxx", "description": "...","score":decimal value which repesent similarity,higher the value represents higher match with user query expected outcome}]
    :return: A list that contian two values: 1. 8-digit HS Code string, 2. product_desc string
    
    """

    context = "\n".join(
        [f"HS Code: {item['hs_code']} | Description: {item['description']} | Score: {item['score']}"
         for item in vector_results]
    )

    prompt = f"""
You are a customs classification expert.

User Product Query:
"{user_query}"

Below are the most relevant HS code matches retrieved from a vector database:

{context}

Task:
- Select the single best matching HS code.
- Return the 8-digit HS code and redifine a proper description based on description of selected hs code.
- Do NOT provide any explanation.
- Strictly follow the output format.

Output format:
[XXXXXXXX,product_desc]

output:defination
XXXXXXXX:hs_code
product_desc:redefine hs code description
"""

    model = genai.GenerativeModel("gemini-3-flash-preview")
    response = model.generate_content(prompt)

    return response
