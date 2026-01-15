import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import httpx
from openai import OpenAI

# --- Load environment variables ---
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE")

# --- Streamlit page config ---
st.set_page_config(
    page_title="📚 AI Study Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 AI Study Assistant ")
st.subheader("Ask questions about Chapter 1 DBA")

# --- Sidebar settings ---
with st.sidebar:
    st.header("Settings ⚙️")
    k = st.slider("Number of context chunks to retrieve", 1, 5, 2)
    show_context = st.checkbox("Show retrieved chunks", value=False)

# --- Load FAISS index and chunks ---
INDEX_PATH = "dba_index.faiss"
CHUNKS_PATH = "chunks.pkl"

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    st.success(f"Loaded FAISS index ({index.ntotal} vectors) and {len(chunks)} chunks.")
else:
    st.error("FAISS index or chunks.pkl not found! Please generate them first.")
    st.stop()

# --- Embedding model ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- RAG retrieval function ---
def query_pdf(question, k=3):
    q_emb = embed_model.encode([question]).astype('float32')
    distances, indices = index.search(q_emb, k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# --- Hosted Token Factory LLM ---
http_client = httpx.Client(verify=False)
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE, http_client=http_client)

def ask_hosted_llm(context, question):
    system_prompt = (
        "Tu es un assistant utile et concis."
        "Utilise uniquement le contexte fourni pour répondre aux questions."
        "Si l'utilisateur pose une question hors contexte, réponds poliment que tu ne peux pas aider."
        "Si l'utilisateur dit des choses comme salut ou bonsoir ou n'importe quelle chose avant d'enter dans le sujet repond naturellement et fluidement  "
    )     
    user_prompt = f"Contexte:\n{context}\n\nQuestion:\n{question}\nRéponse:"

    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=300,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response.choices[0].message.content

# --- Chat history session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- User input ---
user_input = st.chat_input("Type your question here...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Retrieving answer..."):
        # Retrieve context and get AI response
        retrieved_chunks = query_pdf(user_input, k=k)
        context_text = "\n".join(retrieved_chunks)
        ai_response = ask_hosted_llm(context_text, user_input)
    
    # Add AI response to session state
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# --- Display chat messages ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# --- Optionally show retrieved chunks ---
if show_context and st.session_state.messages:
    st.markdown("---")
    st.markdown("### 🔍 Retrieved Context Chunks")
    for i, chunk in enumerate(retrieved_chunks):
        st.markdown(f"**Chunk {i+1}:** {chunk[:500]}...")
