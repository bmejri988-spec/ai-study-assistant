from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# --- Load PDF ---
pdf_reader = PdfReader("DBA_ch1.pdf")

#step1:text extraction from PDF
page_content = [page.extract_text() for page in pdf_reader.pages]

#step2:split text into chunks
# --- Combine and split ---
all_text = " ".join(page_content)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_text(all_text)

#step3: create embeddings and build FAISS index (Turn chunks into vectors)
# 1️⃣ Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, good for RAG

# 2️⃣ Create embeddings for each chunk
embeddings = model.encode(chunks, show_progress_bar=True)

# 3️⃣ Convert embeddings to float32 numpy array (FAISS requirement)
embeddings = np.array(embeddings).astype('float32')

# 4️⃣ Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --- Save index and chunks ---
faiss.write_index(index, "index.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index and chunks saved successfully!")