import google.generativeai as genai
from pinecone import Pinecone
from PyPDF2 import PdfReader
import uuid

# Configure Gemini
genai.configure(api_key="")

# Configure Pinecone
pc = Pinecone(api_key="")
index = pc.Index("hs-code-index")


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_and_store_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = chunk_text(full_text)

    vectors = []

    for chunk in chunks:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk
        )

        embedding = response["embedding"]

        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk
            }
        })

    index.upsert(vectors)
    print("✅ PDF Embedded & Stored Successfully")


# Run this
embed_and_store_pdf("TradeStat-Meidb-Export-Commoditywise-pages-1.pdf")
