# 📚 AI Study Assistant

An AI-powered study assistant that helps you **learn and revise efficiently** by asking questions directly about your lecture notes or textbook chapters. Built with **Python, Streamlit, FAISS, Sentence-Transformers, and a hosted LLM (Llama 3.1)**.

---

## 💡 Inspiration

I’ve always revised subjects using **exams or practice questions**, instead of reading entire chapters. Whenever I didn’t know an answer, I would **go back to the chapter, search for the relevant part, and formulate a response**.  

I realized this could be **automated**. Now, instead of flipping through chapters, I just **provide the PDF(s)** and ask any question — the AI **finds the relevant information and gives a perfectly structured answer**.  

---

## ⚡ Features

- Ask questions about your PDFs or lecture notes  
- Intelligent context retrieval using **FAISS + Sentence-Transformers**  
- Answers generated with **Token Factory hosted LLM (Llama 3.1)**  
- Optional display of retrieved chunks for transparency  
- Adjustable number of context chunks in the sidebar  

---

## 🛠 Tech Stack

- **Python 3.12**  
- **Streamlit** — interactive web app  
- **FAISS** — vector search  
- **Sentence-Transformers** — embeddings  
- **Token Factory hosted LLM** — Llama 3.1  
- **dotenv** — environment variables  
- **PyPDF + LangChain Text Splitter** — PDF ingestion and chunking  

---

## 📥 Setup

1. Clone the repository:

git clone https://github.com/yourusername/ai-study-assistant.git
cd ai-study-assistant

2. Create a virtual environment:

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

3. Install dependencies:

pip install -r requirements.txt

4. Add your API key to `.env`:

LLM_API_KEY=YOUR_HOSTED_LLM_KEY
LLM_API_BASE=https://tokenfactory.esprit.tn/api

5. Place your **precomputed `dba_index.faiss` and `chunks.pkl`** in the root folder.

---

## 📝 Ingesting a PDF

To add a new PDF or update the existing content, use the `ingest.py` script:

✅ To add **another PDF**, just:

1. Change `"pdf_reader = PdfReader("DBA_ch1.pdf")"` to the new file path.  
2. Run `ingest.py`.  
3. The new `chunks.pkl` and `index.faiss` will include the new PDF content.  

---

## 🚀 Usage

Run the app:

streamlit run app.py

- Type a question in the input box  
- Toggle "Show retrieved chunks" to view context  
- Adjust number of chunks via the sidebar  

---

## 📖 How it works

1. **Embedding & Indexing:** PDF text is split into chunks, and embeddings are computed using **Sentence-Transformers**. Chunks are stored in **FAISS**.  
2. **Retrieval:** When a question is asked, the most relevant chunks are retrieved from FAISS.  
3. **Hosted LLM:** The retrieved chunks are fed into **Llama 3.1 hosted on Token Factory**, producing a concise, structured answer.  

---
