from Embeding import search_similar_chunks
from gemini_llm_calls import get_best_hs_code

from pinecone import Pinecone


import google.generativeai as genai
from pinecone import Pinecone


class HSCodeRetriever:

    def __init__(self, gemini_api_key, pinecone_api_key, index_name):
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)

        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)

    
    def get_query_embedding(self, text):
        response = genai.embed_content(
            model="models/text-embedding-004",  # Gemini embedding model
            content=text
        )
        return response["embedding"]

    def fetch_hs_code(self, product_name, product_description):
        # Combine user input
        query = f"Product: {product_name}\nDescription: {product_description}"
        results = search_similar_chunks(query, 3)
        # matches=[]
        # for match in results:
        #     matches.append({
        #         "hs_code": match["metadata"].get("hs_code"),
        #         "description": match["metadata"].get("description"),
        #         "score": match["metadata"].get("score")
        #     })
        user_query=f"What is the HS Code of Product: {product_name} and with Description: {product_description}? "    
        final_hs_code=get_best_hs_code(user_query=user_query,vector_results=results)
        return final_hs_code

        