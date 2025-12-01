import streamlit as st
import pandas as pd
import requests
import os
from typing import Optional
import base64
import time
from pathlib import Path

# ==========================
# Configuration
# ==========================
# Replace with your actual Render backend URL after deployment
#BACKEND_URL = os.getenv("RAG_API_BASE", "https://rag-backend-llnz.onrender.com")
# Try to get backend URL from Streamlit secrets first, then env, then fall back to hard-coded default
DEFAULT_BACKEND_URL = "https://rag-backend-llnz.onrender.com"

try:
    BACKEND_URL = st.secrets.get("RAG_API_BASE", "").strip()
except Exception:
    BACKEND_URL = ""

if not BACKEND_URL:
    BACKEND_URL = os.getenv("RAG_API_BASE", "").strip()

if not BACKEND_URL:
    BACKEND_URL = DEFAULT_BACKEND_URL

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Catch – Unified Story Retriever",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: black;
}
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
}
.block-container { padding-top: 0.4rem; }
.header-wrap {
  display:flex;
  justify-content:left;
  align-items:left;
  flex-direction:column;
  text-align:left;
  overflow-x:hidden;
  margin:0 auto;
  width:100%;
}
.header-divider { height:0.8px; background:#e5e7eb; border:0; margin:6px 0 8px 0; }
</style>
""", unsafe_allow_html=True)

# ==========================
# Load CSV Files & Logo
# ==========================
@st.cache_data(show_spinner=False)
def load_csv(filename):
    """Load CSV file from the assets directory"""
    file_path = Path("assets") / filename
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def encode_image_base64(image_path: str):
    """Encode image to base64"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

# Load CSV files
auth_df = load_csv("authors.csv")
status_df = load_csv("status.csv")

# ==========================
# Header with Logo
# ==========================
st.markdown('<div class="header-wrap">', unsafe_allow_html=True)

LOGO_PNG_PATH = Path("assets/RAG-Catch.png")

if LOGO_PNG_PATH.exists():
    logo_base64 = encode_image_base64(str(LOGO_PNG_PATH))
    if logo_base64:
        logo_and_title_html = f"""
        <div style="display:flex;align-items:center;gap:5px;margin-top:-5px;">
            <img src="data:image/png;base64,{logo_base64}" alt="Catch Logo" style="height:200px;width:200px;">
            <h2 style="margin:0;font-weight:1200;color:#1e3a8a;">Catch – Unified Story Retriever</h2>
        </div>
        """
        st.markdown(logo_and_title_html, unsafe_allow_html=True)
        st.markdown("<p style='color:#6b7280;font-size:19px;margin-top:-70px;margin-left:240px;'>Multiple management tools in. One RAG pipeline out.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='margin:0;font-weight:1000;color:#1e3a8a;'>Catch – Unified Story Retriever</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6b7280;font-size:19px;margin-top:4px;'>Multiple management tools in. One RAG pipeline out.</p>", unsafe_allow_html=True)
else:
    st.markdown("<h1 style='margin:0;font-weight:1000;color:#1e3a8a;'>Catch – Unified Story Retriever</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280;font-size:19px;margin-top:4px;'>Multiple management tools in. One RAG pipeline out.</p>", unsafe_allow_html=True)

st.markdown('<hr class="header-divider">', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Sidebar Controls
# ==========================
st.markdown("""
<style>
section[data-testid="stSidebar"] .block-container { padding-top: 0rem; padding-bottom: 0rem; }
div[data-testid="stSidebar"] label { min-height: 0 !important; padding: 0 !important; margin-bottom: 0rem !important; line-height: 1rem !important; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { margin-bottom: 0.2rem !important; }
.stSelectbox, .stSlider { margin-top: -0.3rem !important; margin-bottom: -0.4rem !important; }
[data-baseweb="slider"] { margin-top: 0.2rem !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🧭 Control Panel")

# Backend Status Check
with st.sidebar:
    st.markdown("### 🔗 Backend Status")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=30)
        if response.status_code == 200:
            st.success("✅ Connected")
            st.sidebar.text(f"Backend: {BACKEND_URL}")
        else:
            st.error("❌ Error")
    except:
        st.error("❌ Offline")
        st.caption("⏳ Backend may be starting...")

st.sidebar.markdown("---")

# Model Selection
st.sidebar.markdown("<h3 style='margin-bottom:0;'>🧩&nbsp;Embedding model</h3>", unsafe_allow_html=True)
embedding_model = st.sidebar.selectbox(
    " ",
    ["intfloat/e5-large-v2", "all-MiniLM-L6-v2"],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("<h3 style='margin-bottom:0;'>🤖&nbsp;LLM model</h3>", unsafe_allow_html=True)
llm_model = st.sidebar.selectbox(
    " ",
    ["meta-llama/Meta-Llama-3-8B-Instruct", "gpt-4o-mini"],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Filters from CSV
authors = ["None"]
statuses = ["None"]

if not auth_df.empty and "author" in auth_df.columns:
    authors = ["None"] + sorted(auth_df["author"].dropna().unique().tolist())

if not status_df.empty and "status" in status_df.columns:
    statuses = ["None"] + sorted(status_df["status"].dropna().unique().tolist())

author_choice = st.sidebar.selectbox("Author", options=authors, index=0)
status_choice = st.sidebar.selectbox("Status", options=statuses, index=0)

author_value = None if author_choice == "None" else author_choice
status_value = None if status_choice == "None" else status_choice

st.sidebar.markdown("---")

# Search Parameters
alpha = st.sidebar.slider("alpha (hybrid weight)", 0.0, 1.0, 0.45, 0.01)
top_k = st.sidebar.slider("Top-K results", 1, 20, 5, 1)

st.sidebar.caption("All controls apply to the next query.")

# ==========================
# Helper Functions
# ==========================
@st.cache_resource(show_spinner=False)
def get_http_session():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s

def wake_up_backend():
    """Wake up the backend if it's sleeping (Render free tier)"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=60)
        return response.status_code == 200
    except:
        return False

if not wake_up_backend():
    st.warning("⏳ Backend is waking up (this takes ~60 seconds on free tier)...")
    time.sleep(60) 
    
def run_rag_pipeline(query: str, embedding_model: str, llm_model: str,
                     author: Optional[str], status: Optional[str],
                     alpha: float, k: int) -> dict:
    """Call the FastAPI backend /ask endpoint"""
    
    if not BACKEND_URL or BACKEND_URL == "https://rag-backend-llnz.onrender.com":
        return {
            "answer": "⚠️ Backend URL not configured. Please set RAG_API_BASE environment variable.",
            "chunks": []
        }
    
    filters = None
    parts = []
    if author:
        parts.append({"Author": author})
    if status:
        parts.append({"Status": status})
    
    if len(parts) == 1:
        filters = parts[0]
    elif len(parts) > 1:
        filters = {"$and": parts}
    
    payload = {
        "query": query,
        "top_k": int(k),
        "alpha": float(alpha),
        "filters": filters,
        "embed_model": embedding_model,
        "llm_model": llm_model,
    }
    
    try:
        session = get_http_session()
        resp = session.post(f"{BACKEND_URL}/ask", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        
        answer_text = data.get("answer", "No answer returned")
        chunks = data.get("chunks", [])
        
        return {"answer": answer_text, "chunks": chunks}
        
    except requests.HTTPError as e:
        error_detail = getattr(e.response, 'text', str(e))
        return {
            "answer": f"⚠️ Backend HTTP error: {e}\n\nDetails: {error_detail}",
            "chunks": []
        }
    except requests.RequestException as e:
        return {
            "answer": f"⚠️ Cannot reach backend at {BACKEND_URL}: {e}",
            "chunks": []
        }
    except Exception as e:
        return {
            "answer": f"⚠️ Unexpected error: {e}",
            "chunks": []
        }

# ==========================
# Main Query Interface
# ==========================
st.subheader("Your query")
st.caption("Use the sidebar to configure models and filters. Press **Run** to fetch a response.")

with st.form("query_form", clear_on_submit=False):
    query = st.text_input(
        " ",
        placeholder="e.g., What user stories mention drag-and-drop uploads?",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Run")

if submitted:
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        with st.spinner("Running RAG pipeline..."):
            result = run_rag_pipeline(
                query=query.strip(),
                embedding_model=embedding_model,
                llm_model=llm_model,
                author=author_value,
                status=status_value,
                alpha=alpha,
                k=top_k
            )
        
        # Display Answer
        st.markdown("### 🔎 Answer")
        st.write(result["answer"])
        
        # Display Retrieved Chunks
        st.markdown("### 📚 Retrieved Chunks")
        chunks = result.get("chunks", [])
        
        if chunks:
            for ch in chunks:
                with st.container(border=True):
                    st.markdown(f"**Chunk ID:** {ch.get('id', 'N/A')}")
                    st.markdown(f"**Topic:** {ch.get('topic', 'N/A')}")
                    st.markdown(f"**Author:** {ch.get('author', 'N/A')}")
                    st.markdown(f"**Status:** {ch.get('status', 'N/A')}")
                    st.markdown(f"**Score:** {ch.get('score', 0):.2f}")
                    st.markdown(f"**Snippet:** {ch.get('combined_text', '')[:500]}...")
        else:
            st.info("ℹ️ No chunks retrieved.")
