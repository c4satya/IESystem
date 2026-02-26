import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import json
from Embeding import *

# load_dotenv()

# GEMINI_API_KEY = "AIzaSyD7TeAVbKHOpJsNURIc64H8VBX0pFQGF6s"



# Configure Gemini


def get_best_hs_code(user_query: str, vector_results):
   GEMINI_API_KEY = "AIzaSyD7TeAVbKHOpJsNURIc64H8VBX0pFQGF6s"
   genai.configure(api_key=GEMINI_API_KEY)
   
   formatted_chunks = []
   if isinstance(vector_results[0], str):
    # vector_results are raw strings
    formatted_chunks = [f"Chunk:\n{chunk}\n" for chunk in vector_results]
   else:
    # vector_results are dicts with metadata
        formatted_chunks = [f"Chunk {result['id']}:\n{result['metadata']['text']}\n" for result in vector_results]
    
   prompt = f"""You are a customs tariff classification expert with 20+ years experience.

USER QUERY: "{user_query}"

TOP VECTOR MATCHES (raw export data):
{formatted_chunks}

RULES:
1. Parse each chunk using this format:
   - After \\n → 4 digits = SEQ NO
   - Next 8 digits = HS CODE  
   - Next text until numbers = DESCRIPTION
   - Ignore all trailing metrics/numbers

2. From all valid 8-digit HS codes found, select EXACTLY ONE best match for the user query
3. Use your HS code knowledge to validate/correct if needed
4. Write a clean, professional product description (max 80 chars)

OUTPUT JSON ONLY - NO OTHER TEXT:
{{"hs_code": "XXXXXXXX", "product_desc": "Clean description here"}}

EXAMPLE OUTPUT:
{{"hs_code": "68022310", "product_desc": "Polished granite blocks/tiles"}}
"""
    

   model = genai.GenerativeModel("gemini-3-flash-preview")
   response = model.generate_content(prompt)
   

   return response.text


