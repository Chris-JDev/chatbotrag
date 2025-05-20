import os
import re
import json
import datetime
import tempfile
import shutil
from typing import List, Dict, Any, Optional

import streamlit as st
import streamlit.components.v1 as components
# Attempt Lottie support; if missing, stub it out
try:
    from streamlit_lottie import st_lottie
except ImportError:
    def st_lottie(*args, **kwargs):
        pass
import requests

# --- New: Rule-based classifier patterns from PDF methodology ---
CAMPAIGN_PATTERNS = {
    "Social Media Marketing": [r"social media", r"facebook", r"instagram", r"twitter", r"linkedin"],
    "Content Marketing": [r"blog", r"article", r"content strategy", r"seo content"],
    "Email Marketing": [r"email", r"newsletter", r"mail campaign"],
    "SEO": [r"seo", r"search engine", r"ranking"],
    "PPC Advertising": [r"ppc", r"pay[- ]?per[- ]?click", r"google ads", r"adwords"],
    "Influencer Marketing": [r"influencer", r"celebrity endorsement", r"collaborat"],
    "Video Marketing": [r"video", r"youtube", r"tiktok", r"video content"]
}

MARKETING_CATEGORIES = list(CAMPAIGN_PATTERNS.keys())
CATEGORY_DESCRIPTIONS = {
    "Social Media Marketing": "Strategies for Facebook, Instagram, Twitter and LinkedIn",
    "Content Marketing": "Blogs, articles, and content strategy",
    "Email Marketing": "Newsletter and email campaign optimization",
    "SEO": "Search engine optimization techniques",
    "PPC Advertising": "Pay-per-click campaign management",
    "Influencer Marketing": "Working with content creators and influencers",
    "Video Marketing": "YouTube, TikTok and video content strategies"
}

# Import LangChain components
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, Docx2txtLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ---------- Core Configuration ----------
OLLAMA_BASE_URL = os.getenv(
    "https://40e6-91-73-226-54.ngrok-free.app", "http://localhost:11434"
)

# Load Lottie animations for richer UI
@st.cache_data(show_spinner=False)
def load_lottieurl(url: str) -> Optional[Dict]:
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# ---------- Core Helpers ----------
@st.cache_resource(show_spinner=False)
def get_ollama_client(model: str, temp: float = 0.3):
    return Ollama(base_url=OLLAMA_BASE_URL, model=model, temperature=temp)

@st.cache_data(show_spinner=False)
def process_documents(docs_list):
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for file in docs_list:
            p = os.path.join(td, file.name)
            with open(p, "wb") as f:
                f.write(file.getbuffer())
            paths.append(p)
        texts = []
        for p in paths:
            try:
                if p.endswith(".pdf"):
                    loader = PDFPlumberLoader(p)
                elif p.endswith(".docx"):
                    loader = Docx2txtLoader(p)
                else:
                    loader = TextLoader(p)
                texts.extend(loader.load())
            except Exception as e:
                st.error(f"Load error {os.path.basename(p)}: {e}")
        if not texts:
            st.error("No documents loaded.")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        chunks = splitter.split_documents(texts)
        if not chunks:
            st.error("No chunks generated.")
            return None

        dir_ = "./marketing_db"
        if os.path.exists(dir_):
            shutil.rmtree(
                dir_, onerror=lambda func, path, exc: (
                    os.chmod(path, 0o777), func(path)
                )
            )

        # iterate embedding models
        for model in ["nomic-embed-text", "all-MiniLM", "llama2"]:
            try:
                st.info(f"Embedding with {model}…")
                emb = OllamaEmbeddings(
                    base_url=OLLAMA_BASE_URL, model=model
                )
                if not emb.embed_query(
                    chunks[0].page_content[:30]
                ):
                    st.warning(f"{model} gave empty embedding")
                    continue
                vs = Chroma.from_documents(
                    chunks, emb, persist_directory=dir_
                )
                st.success(f"Stored vectors with {model}")
                return vs
            except Exception as e:
                st.warning(f"{model} failed: {e}")
        st.error("All embedding models failed.")
        return None

# Build retriever from vector store
def get_retriever():
    vs = st.session_state.get("vector_store")
    if not vs:
        return None
    return vs.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 8}
    )

# Create AIDA-aware prompt template
def get_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a marketing QA system. Use only provided docs and structure answers using the AIDA marketing model (Attention, Interest, Desire, Action)."
        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\nQuestion: {question}"
        ),
    ])

# Rule-based campaign classifier
def classify_campaign(text: str) -> Optional[str]:
    text_lower = text.lower()
    for campaign, patterns in CAMPAIGN_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return campaign
    return None

# Initialize QA chain
def init_qa_chain():
    if (
        not st.session_state.get("qa_chain")
        and st.session_state.get("vector_store")
    ):
        try:
            llm = get_ollama_client("llama2", 0.1)
            retr = get_retriever()
            if retr:
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retr,
                    chain_type_kwargs={"prompt": get_prompt()},
                )
                st.session_state.qa_chain = qa
                return qa
        except Exception as e:
            st.error(f"QA init error: {e}")
    return st.session_state.get("qa_chain")

# Simple formatting of responses
def format_resp(r):
    return getattr(r, "content", str(r)).strip()

# Save & load chat history
# ... (unchanged) ...

# Initialize session state defaults
def init_state():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("vector_store", None)
    ss.setdefault("qa_chain", None)
    ss.setdefault("llm", get_ollama_client("llama2", 0.3))
    ss.setdefault("selected_category", MARKETING_CATEGORIES[0])
    ss.setdefault("chat_started", False)
    ss.setdefault(
        "available_histories",
        [
            f for f in os.listdir("chat_histories") if f.endswith(".json")
        ]
        if os.path.exists("chat_histories")
        else [],
    )

# ---------- Streamlit App ----------
def main():
    st.set_page_config(
        page_title="Marketing Advisor",
        page_icon="📊",
        layout="wide",
    )
    init_state()

    with st.sidebar:
        st.title("Marketing Advisor")
        # Lottie introduction animation
        # ... (unchanged) ...

        # Document upload
        st.header("Upload Resources")
        # ... (unchanged) ...

        st.markdown("---")
        st.header("Rule-Based Category Classifier")
        st.info("Optionally enter a campaign-related statement to auto-select a category.")
        user_sample = st.text_area("Classifier Input (optional)", height=80)
        if st.button("Classify Campaign"):
            camp = classify_campaign(user_sample)
            if camp:
                st.session_state.selected_category = camp
                st.success(f"Detected category: {camp}")
            else:
                st.error("Could not detect a category. Please refine your input.")

        st.markdown("---")
        # Quick idea generator and chat management
        # ... (unchanged) ...

    # Main area
    st.title(f"Marketing Advisor: {st.session_state.selected_category}")
    if not st.session_state.chat_started:
        st.info("Use the sidebar to upload docs or start chatting.")
        if st.button("Start Chat"):
            st.session_state.chat_started = True
            st.experimental_rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if ui := st.chat_input("Ask your marketing question…"):
        st.session_state.chat_started = True
        st.session_state.messages.append({"role": "user", "content": ui})
        # Attempt auto-classification
        auto_cat = classify_campaign(ui)
        if auto_cat:
            st.session_state.selected_category = auto_cat
        if st.session_state.vector_store:
            qa = init_qa_chain()
            ans = qa.run(ui) if qa else "Error: QA unavailable"
        else:
            # Fallback generative
            prompt = f"Using AIDA, generate a marketing {st.session_state.selected_category} plan for: {ui}."
            r = st.session_state.llm.invoke(prompt)
            ans = getattr(r, "content", str(r))
        fr = format_resp(ans)
        st.session_state.messages.append({"role": "assistant", "content": fr})
        st.chat_message("assistant").markdown(fr)

if __name__ == "__main__":
    main()
