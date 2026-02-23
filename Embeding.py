from dotenv import load_dotenv
from pypdf import PdfReader
import uuid
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pdf-embeddings"
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# 1️⃣ Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# 2️⃣ Chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 3️⃣ Create embeddings
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings

# 4️⃣ Upload to Pinecone
def upload_to_pinecone(chunks, embeddings):
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding.tolist(),
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)

# 🚀 Full Pipeline
if __name__ == "__main__":
    #pdf_path1 = r"C:\Users\Atik Dighe\Downloads\IESystem-main\IESystem-main\DATA\TradeStat-Meidb-Export-Commoditywise-pages-1.pdf"
    pdf_path2= r"C:\Users\Atik Dighe\Downloads\IESystem-main\IESystem-main\DATA\TradeStat-Meidb-Export-Commoditywise-pages-2.pdf"
    pdf_path3= r"C:\Users\Atik Dighe\Downloads\IESystem-main\IESystem-main\DATA\TradeStat-Meidb-Export-Commoditywise-pages-3.pdf"
    pdf_path4= r"C:\Users\Atik Dighe\Downloads\IESystem-main\IESystem-main\DATA\TradeStat-Meidb-Export-Commoditywise-pages-4.pdf"
    
    print("Extracting text...")
   # text1 = extract_text_from_pdf(pdf_path1)
    text2= extract_text_from_pdf(pdf_path2)
    text3 = extract_text_from_pdf(pdf_path3)
    text4= extract_text_from_pdf(pdf_path4)
    


    print("Chunking...")
    chunks2 = chunk_text(text2)
    chunks3 = chunk_text(text3)
    chunks4 = chunk_text(text4)
    

    print("Creating embeddings...")
    embeddings2= embed_chunks(chunks2)
    embeddings3 = embed_chunks(chunks3)
    embeddings4 = embed_chunks(chunks4)


    print("Uploading to Pinecone...")
    upload_to_pinecone(chunks2, embeddings2)
    upload_to_pinecone(chunks3, embeddings3)
    upload_to_pinecone(chunks4, embeddings4)

    print("✅ PDF successfully embedded into Pinecone.")