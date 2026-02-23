from openai import OpenAI
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

    models = genai.list_models()
    for m in models:
        print(m.name)
    def get_query_embedding(self, text):
        response = genai.embed_content(
            model="models/text-embedding-004",  # Gemini embedding model
            content=text
        )
        return response["embedding"]

    def fetch_hs_code(self, product_name, product_description, top_k=5):
        # Combine user input
        query_text = f"Product: {product_name}\nDescription: {product_description}"

        # Generate embedding
        query_vector = self.get_query_embedding(query_text)

        # Query Pinecone
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        # Format results
        matches = []
        for match in results["matches"]:
            matches.append({
                "hs_code": match["metadata"].get("hs_code"),
                "description": match["metadata"].get("description"),
                "score": match["score"]
            })

        return matches
        