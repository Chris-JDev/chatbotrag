import os
import json
import datetime
import tempfile
import shutil
from typing import List, Dict, Any, Optional

import streamlit as st
from pyngrok import ngrok
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, Docx2txtLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ✅ Set to public ngrok URL for local Ollama
OLLAMA_BASE_URL = "https://e179-5-32-57-218.ngrok-free.app"

MARKETING_CATEGORIES = [
    "Digital Marketing", "Content Marketing", "Social Media Marketing",
    "Email Marketing", "SEO & SEM", "Influencer Marketing",
    "Brand Development", "Market Research"
]

CATEGORY_DESCRIPTIONS = {
    "Digital Marketing": "Strategies for online marketing channels including websites, apps, social media, email, and search engines.",
    "Content Marketing": "Creating and distributing valuable content to attract and engage a target audience.",
    "Social Media Marketing": "Strategies specific to social platforms like Instagram, Facebook, LinkedIn, Twitter, and TikTok.",
    "Email Marketing": "Direct marketing strategies using email to promote products or services.",
    "SEO & SEM": "Techniques to improve search engine visibility and paid search strategies.",
    "Influencer Marketing": "Partnering with influential people to increase brand awareness or drive sales.",
    "Brand Development": "Strategies to build, strengthen and promote a company's brand identity.",
    "Market Research": "Methods to gather and analyze information about consumers, competitors, and market trends."
}

def check_ollama_connection():
    try:
        client = ChatOllama(base_url=OLLAMA_BASE_URL, model="llama2", temperature=0.3)
        r = client.invoke("ping")
        return True, getattr(r, "content", str(r))
    except Exception as e:
        return False, str(e)

def process_documents(docs_list):
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for file in docs_list:
            p = os.path.join(td, file.name)
            with open(p, "wb") as f: f.write(file.getbuffer())
            paths.append(p)
        texts = []
        for p in paths:
            try:
                if p.endswith(".pdf"): loader = PDFPlumberLoader(p)
                elif p.endswith(".docx"): loader = Docx2txtLoader(p)
                else: loader = TextLoader(p)
                texts.extend(loader.load())
            except Exception as e:
                st.error(f"Load error {os.path.basename(p)}: {e}")
        if not texts:
            st.error("No documents loaded."); return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(texts)
        if not chunks:
            st.error("No chunks generated."); return None

        dir_ = "./marketing_db"
        if os.path.exists(dir_):
            shutil.rmtree(dir_, onerror=lambda func, path, exc: (os.chmod(path, 0o777), func(path)))

        for model in ["nomic-embed-text", "all-MiniLM", "llama2"]:
            try:
                st.info(f"Embedding with {model}…")
                emb = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=model)
                if not emb.embed_query(chunks[0].page_content[:30]):
                    st.warning(f"{model} gave empty embedding"); continue
                vs = Chroma.from_documents(chunks, emb, persist_directory=dir_)
                st.success(f"Stored vectors with {model}")
                return vs
            except Exception as e:
                st.warning(f"{model} failed: {e}")
        st.error("All embedding models failed.")
        return None

def get_retriever():
    if not st.session_state.vector_store: return None
    return st.session_state.vector_store.as_retriever(search_type="mmr",
                                                      search_kwargs={"k":6,"fetch_k":8})

def get_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a marketing QA system. Use only provided docs."),
        HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
    ])

def init_qa_chain():
    if not st.session_state.qa_chain and st.session_state.vector_store:
        try:
            llm = ChatOllama(base_url=OLLAMA_BASE_URL, model="llama2", temperature=0.1)
            retr = get_retriever()
            if retr:
                qa = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=retr,
                    chain_type_kwargs={"prompt": get_prompt()}
                )
                st.session_state.qa_chain = qa
                return qa
        except Exception as e:
            st.error(f"QA init error: {e}")
    return st.session_state.qa_chain

def format_resp(r):
    return getattr(r, "content", str(r)).strip()

def save_history(msgs, fname=None):
    os.makedirs("chat_histories", exist_ok=True)
    if not fname: fname = f"chat_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    path = os.path.join("chat_histories", fname)
    try:
        with open(path,"w") as f: json.dump(msgs, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Save error: {e}"); return False

def load_history(fname):
    try:
        with open(os.path.join("chat_histories", fname)) as f: return json.load(f)
    except Exception as e:
        st.error(f"Load error: {e}"); return None

def init_state():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("vector_store", None)
    ss.setdefault("qa_chain", None)
    ss.setdefault("llm", ChatOllama(base_url=OLLAMA_BASE_URL, model="llama2", temperature=0.3))
    ss.setdefault("selected_category", MARKETING_CATEGORIES[0])
    ss.setdefault("chat_started", False)
    ss.setdefault("available_histories",
                  [f for f in os.listdir("chat_histories") if f.endswith(".json")]
                  if os.path.exists("chat_histories") else [])

def main():
    st.set_page_config(page_title="Marketing Advisor", page_icon="📊", layout="wide")
    init_state()

    with st.sidebar:
        st.title("Marketing Advisor")
        cat = st.selectbox("Focus area", MARKETING_CATEGORIES,
                           index=MARKETING_CATEGORIES.index(st.session_state.selected_category))
        if cat != st.session_state.selected_category:
            st.session_state.selected_category = cat
        st.info(CATEGORY_DESCRIPTIONS[cat])
        st.markdown("---")

        st.header("Upload Resources")
        files = st.file_uploader("Upload marketing docs (PDF/DOCX/TXT)",
                                 type=["pdf","docx","txt"], accept_multiple_files=True)
        if files and st.button("Create Knowledge Base"):
            ok,msg = check_ollama_connection()
            if not ok: st.error(f"Ollama error: {msg}")
            else:
                vs = process_documents(files)
                if vs:
                    st.session_state.vector_store = vs
                    st.session_state.qa_chain = None
                    st.success("Knowledge base created!")
        st.markdown("---")

        st.header("Quick Idea Generator")
        if st.button("Generate Quick Ideas"):
            with st.spinner("Generating…"):
                if st.session_state.vector_store:
                    retr = get_retriever()
                    docs = retr.get_relevant_documents(f"Generate 5 ideas for {st.session_state.selected_category}")
                    ctx = "\n\n".join(d.page_content for d in docs)
                    prompt = (f"Based on these docs:\n{ctx}\n"
                              f"Generate 5 quick marketing ideas for {st.session_state.selected_category}. "
                              "For each: headline + 1-sentence explanation.")
                    r = st.session_state.llm.invoke(prompt)
                    ideas = getattr(r, "content", str(r))
                else:
                    r = st.session_state.llm.invoke(f"List 5 quick marketing ideas for {st.session_state.selected_category}.")
                    ideas = getattr(r, "content", str(r))
                st.session_state.messages.append({"role":"assistant","content":ideas})
        st.markdown("---")

        st.header("Chat Management")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("Save Chat"):
                if save_history(st.session_state.messages): st.success("Saved!")
        with c2:
            if st.button("Clear Chat"):
                st.session_state.messages=[]; st.success("Cleared!")

        if st.session_state.available_histories:
            h = st.selectbox("Load chat", [""]+st.session_state.available_histories)
            if h and st.button("Load Selected Chat"):
                msgs = load_history(h)
                if msgs:
                    st.session_state.messages=msgs; st.success("Loaded!")

    st.title(f"Marketing Advisor: {st.session_state.selected_category}")

    if not st.session_state.chat_started:
        st.info("Use sidebar to upload docs or start chat.")
        if st.button("Start Chat"):
            st.session_state.chat_started=True; st.experimental_rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if ui := st.chat_input("Ask your marketing question…"):
        st.session_state.chat_started = True
        st.session_state.messages.append({"role":"user","content":ui})
        if st.session_state.vector_store:
            qa = init_qa_chain()
            ans = qa.run(ui) if qa else "Error: QA unavailable"
        else:
            ans = ChatOllama(base_url=OLLAMA_BASE_URL, model="llama2", temperature=0.3).invoke(ui).content
        fr = format_resp(ans)
        st.session_state.messages.append({"role":"assistant","content":fr})
        st.chat_message("assistant").markdown(fr)

if __name__=="__main__":
    main()
