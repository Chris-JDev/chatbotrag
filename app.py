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
OLLAMA_BASE_URL = "https://7969-91-73-226-54.ngrok-free.app"

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

# ------------------ Task List Functions ------------------
def init_task_list():
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []

def add_task(task: str):
    st.session_state.tasks.append({'task': task, 'completed': False})

def toggle_task(index: int):
    st.session_state.tasks[index]['completed'] = not st.session_state.tasks[index]['completed']

def delete_task(index: int):
    st.session_state.tasks.pop(index)


def task_manager_ui():
    st.sidebar.header("Task Manager")
    new_task = st.sidebar.text_input("Add a new task")
    if st.sidebar.button("Add Task") and new_task:
        add_task(new_task)
    if st.sidebar.button("Clear All Tasks"):
        st.session_state.tasks.clear()

    st.sidebar.markdown("### Your Tasks")
    for idx, task in enumerate(st.session_state.tasks):
        cols = st.sidebar.columns([0.05, 0.7, 0.25])
        if cols[0].checkbox("", value=task['completed'], key=f"completed_{idx}"):
            toggle_task(idx)
        cols[1].markdown(f"~~{task['task']}~~" if task['completed'] else task['task'])
        if cols[2].button("Delete", key=f"delete_{idx}"):
            delete_task(idx)

# ---------------------------------------------------------

def check_ollama_connection():
    try:
        client = ChatOllama(base_url=OLLAMA_BASE_URL, model="llama2", temperature=0.3)
        r = client.invoke("ping")
        return True, getattr(r, "content", str(r))
    except Exception as e:
        return False, str(e)

# ... [other helper functions unchanged] ...

def process_documents(docs_list):
    # existing implementation
    ...

def get_retriever():
    # existing implementation
    ...

def get_prompt():
    # existing implementation
    ...

def init_qa_chain():
    # existing implementation
    ...

def format_resp(r):
    # existing implementation
    ...

def save_history(msgs, fname=None):
    # existing implementation
    ...

def load_history(fname):
    # existing implementation
    ...

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
    init_task_list()

    # Task manager in sidebar
    task_manager_ui()

    # Existing sidebar UI
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
                # existing implementation...
                ...

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

    # Main area
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

if __name__ == "__main__":
    main()
