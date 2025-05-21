import os
import json
import datetime
import tempfile
import shutil
from typing import Optional, Dict, Any

import streamlit as st
import requests
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, Docx2txtLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ─── Setup ───────────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title='Marketing Advisor', page_icon='📊', layout='wide')

# Customize CSS for response font
st.markdown(
    """
    <style>
    .assistant {font-size:20px; color:#fff; margin: 8px 0;}
    .user {font-size:18px; font-weight:bold; color:#eee; margin: 8px 0;}
    </style>
    """, unsafe_allow_html=True
)

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
MARKETING_CATEGORIES = [
    "Digital Marketing", "Content Marketing", "Social Media Marketing",
    "Email Marketing", "SEO & SEM", "Influencer Marketing",
    "Brand Development", "Market Research"
]
CATEGORY_DESCRIPTIONS = {
    "Digital Marketing":      "Strategies for online marketing channels including websites, apps, social media, email, and search engines.",
    "Content Marketing":      "Creating and distributing valuable content to attract and engage a target audience.",
    "Social Media Marketing": "Strategies specific to social platforms like Instagram, Facebook, LinkedIn, Twitter, and TikTok.",
    "Email Marketing":        "Direct marketing strategies using email to promote products or services.",
    "SEO & SEM":              "Techniques to improve search engine visibility and paid search strategies.",
    "Influencer Marketing":   "Partnering with influencers to boost brand awareness or drive sales.",
    "Brand Development":      "Strategies to build, strengthen, and promote a company's brand identity.",
    "Market Research":        "Methods to gather and analyze information about consumers, competitors, and market trends."
}
# AIDA description
AIDA_DESCRIPTION = (
    "**AIDA Marketing Model**\n"
    "1. **Attention**: Capture awareness with compelling hooks or visuals.\n"
    "2. **Interest**: Maintain curiosity by highlighting benefits and features.\n"
    "3. **Desire**: Build emotional engagement through value propositions and social proof.\n"
    "4. **Action**: Prompt a clear next step such as purchase, sign-up, or inquiry.\n\n"
    "Use this framework to guide prospects from awareness to conversion."
)

# ─── Helpers ─────────────────────────────────────────────────────────────────────
def check_ollama_connection() -> (bool, str):
    try:
        client = ChatOllama(base_url=OLLAMA_BASE_URL, model="llama2", temperature=0.3)
        r = client.invoke("ping")
        return True, getattr(r, "content", str(r))
    except Exception as e:
        return False, str(e)

@st.cache_resource(show_spinner=False)
def get_ollama_client(model: str = "llama2", temp: float = 0.3):
    return ChatOllama(base_url=OLLAMA_BASE_URL, model=model, temperature=temp)

@st.cache_resource(show_spinner=False)
def process_documents(files):
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for f in files:
            p = os.path.join(td, f.name)
            with open(p, 'wb') as out:
                out.write(f.getbuffer())
            paths.append(p)
        texts = []
        for p in paths:
            try:
                loader = PDFPlumberLoader(p) if p.endswith('.pdf') else Docx2txtLoader(p) if p.endswith('.docx') else TextLoader(p)
                texts.extend(loader.load())
            except Exception as e:
                st.error(f"Load error {os.path.basename(p)}: {e}")
        if not texts:
            st.error("No documents loaded."); return None
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(texts)
        if not chunks:
            st.error("No chunks generated."); return None
        dbdir = './marketing_db'
        if os.path.exists(dbdir): shutil.rmtree(dbdir, onerror=lambda fn,path,exc:(os.chmod(path,0o777),fn(path)))
        models = ['nomic-embed-text','all-MiniLM','llama2']
        prog = st.progress(0)
        for i,m in enumerate(models,1):
            st.info(f"Embedding with {m} ({i}/{len(models)})…")
            try:
                emb = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=m)
                if not emb.embed_query(chunks[0].page_content[:30]):
                    st.warning(f"{m} returned empty embedding; skipping"); continue
                store = Chroma.from_documents(chunks,emb,persist_directory=dbdir)
                st.success(f"Vectors stored with {m}"); return store
            except Exception as e:
                st.warning(f"{m} failed: {e}")
            prog.progress(i/len(models))
        st.error("All embedding models failed."); return None

def get_retriever():
    vs=st.session_state.vector_store
    return vs.as_retriever(search_type='mmr',search_kwargs={'k':6,'fetch_k':8}) if vs else None

def get_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a marketing QA system. Use only provided docs and structure answers with AIDA."),
        HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
    ])

def init_qa_chain():
    if not st.session_state.qa_chain and st.session_state.vector_store:
        try:
            llm=get_ollama_client('llama2',0.1)
            retr=get_retriever()
            if retr:
                qa=RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=retr,chain_type_kwargs={'prompt':get_prompt()})
                st.session_state.qa_chain=qa; return qa
        except Exception as e:
            st.error(f"QA init error: {e}")
    return st.session_state.qa_chain

def format_resp(r): return getattr(r,'content',str(r)).strip()

def save_history(msgs,fname=None):
    os.makedirs('chat_histories',exist_ok=True)
    if not fname: fname=f"chat_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    path=os.path.join('chat_histories',fname)
    try:
        with open(path,'w') as f: json.dump(msgs,f,indent=2)
        return True
    except Exception as e:
        st.error(f"Save error: {e}"); return False

def load_history(fname):
    try: return json.load(open(os.path.join('chat_histories',fname)))
    except Exception as e: st.error(f"Load error: {e}"); return None

def init_state():
    ss=st.session_state
    ss.setdefault('messages',[])
    ss.setdefault('vector_store',None)
    ss.setdefault('qa_chain',None)
    ss.setdefault('llm',get_ollama_client('llama2',0.3))
    ss.setdefault('selected_category',MARKETING_CATEGORIES[0])
    ss.setdefault('chat_started',False)
    ss.setdefault('available_histories',[f for f in os.listdir('chat_histories') if f.endswith('.json')] if os.path.exists('chat_histories') else [])

def main():
    init_state()
    with st.sidebar:
        st.title('Marketing Advisor')
        cat=st.selectbox('Focus area',MARKETING_CATEGORIES,index=MARKETING_CATEGORIES.index(st.session_state.selected_category))
        if cat!=st.session_state.selected_category: st.session_state.selected_category=cat
        st.info(CATEGORY_DESCRIPTIONS[cat]); st.markdown('---')
        if st.button('Check Ollama Connection',key='conn'): ok,msg=check_ollama_connection(); st.success('Connected: '+msg) if ok else st.error('Error: '+msg)
        st.markdown('---')
        st.header('Upload Resources')
        files=st.file_uploader('Upload docs (PDF/DOCX/TXT)',type=['pdf','docx','txt'],accept_multiple_files=True)
        if files and st.button('Create Knowledge Base',key='create_kb'):
            vs=process_documents(files)
            if vs: st.session_state.vector_store=vs; st.session_state.qa_chain=None; st.success('Knowledge base created!')
        st.markdown('---')
        st.header('Quick Idea Generator')
        if st.button('Generate Quick Ideas',key='quick_ideas'):
            if not st.session_state.vector_store: st.error('Please upload docs first.')
            else:
                with st.spinner('Generating…'):
                    retr=get_retriever(); docs=retr.get_relevant_documents(f'Generate 5 marketing ideas for {st.session_state.selected_category}')
                    ctx='\n\n'.join(d.page_content for d in docs)
                    prompt=f'Context:\n{ctx}\n\nQuestion: Generate 5 quick marketing ideas for {st.session_state.selected_category}. For each: headline + 1-sentence explanation.'
                    r=st.session_state.llm.invoke(prompt); ideas=getattr(r,'content',str(r))
                st.session_state.messages.append({'role':'assistant','content':ideas})
        st.markdown('---')
        st.markdown('### AIDA Model Explanation')
        if st.button('Show AIDA Explanation',key='aida_explain'): st.session_state.messages.append({'role':'assistant','content':AIDA_DESCRIPTION})
        st.markdown('---')
        st.markdown('### AIDA-Driven Plan')
        if st.button('Generate AIDA Plan',key='aida_plan'):
            if not st.session_state.vector_store: st.error('Please upload docs first.')
            else:
                with st.spinner('Generating AIDA plan…'):
                    retr=get_retriever(); docs=retr.get_relevant_documents(f'Generate an AIDA-structured marketing plan for {st.session_state.selected_category}')
                    ctx='\n\n'.join(d.page_content for d in docs)
                    prompt=f'Context:\n{ctx}\n\nQuestion: Generate a full marketing plan for {st.session_state.selected_category}, structured according to the AIDA model.'
                    r=st.session_state.llm.invoke(prompt); plan=getattr(r,'content',str(r))
                st.session_state.messages.append({'role':'assistant','content':plan})
        st.markdown('---')
        st.header('Chat Management')
        c1,c2=st.columns(2)
        with c1:
            if st.button('Save Chat',key='save_chat'): save_history(st.session_state.messages) and st.success('Saved!')
        with c2:
            if st.button('Clear Chat',key='clear_chat'): st.session_state.messages=[]; st.success('Cleared!')
        if st.session_state.available_histories:
            h=st.selectbox('Load chat',['']+st.session_state.available_histories,key='load_sel')
            if h and st.button('Load Selected Chat',key='load_chat'): msgs=load_history(h); msgs and setattr(st.session_state,'messages',msgs) and st.success('Loaded!')
    # Main area
    st.title(f'Marketing Advisor: {st.session_state.selected_category}')
    if not st.session_state.chat_started:
        st.info('Use the sidebar and start chatting.')
        if st.button('Start Chat',key='start_chat'): st.session_state.chat_started=True
    # Render messages in right panel
    for m in st.session_state.messages:
        cls='assistant' if m['role']=='assistant' else 'user'
        st.markdown(f"<div class='{cls}'>{m['role'].title()}: {m['content']}</div>",unsafe_allow_html=True)
    # Chat input
    if st.session_state.chat_started:
        if ui:=st.chat_input('Ask your marketing question…',key='chat_input'):
            st.session_state.messages.append({'role':'user','content':ui})
            qa=init_qa_chain()
            ans=qa.run(ui) if qa else getattr(st.session_state.llm.invoke(ui),'content',str(ui))
            st.session_state.messages.append({'role':'assistant','content':ans})
            st.markdown(f"<div class='assistant'>Assistant: {ans}</div>",unsafe_allow_html=True)

if __name__=='__main__':
    main()
