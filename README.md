# ChatbotRAG

A Streamlit-based **Marketing Advisor** that uses Retrieval-Augmented Generation (RAG) over your own documents plus the AIDA framework, powered by a self-hosted Ollama LLM (forwarded via ngrok or hosted on Render.com) and a Chroma vector store.

![Example UI](./screenshot.png)

---

## 🚀 Features

- **Custom Knowledge Base**  
  Upload PDF / DOCX / TXT files to build a domain-specific corpus.

- **Quick Idea Generator**  
  Instantly generate **5 marketing ideas** (headline + 1-sentence explanation) based on your docs.

- **AIDA Framework**  
  - **Explanation**: Learn the four steps—Attention, Interest, Desire, Action.  
  - **Plan**: Generate a full marketing plan structured by AIDA.

- **Chat Interface**  
  Ask free-form marketing questions answered from your uploaded content.

---

## 📁 Repo Structure

.
├── .gitignore
├── Procfile # for Render.com deployment
├── app.py # main Streamlit application
├── requirements.txt # Python dependencies
└── README.md # this file

markdown
Copy
Edit

- **`app.py`**  
  Entry point for the Streamlit app.
- **`requirements.txt`**  
  All Python packages required (Streamlit, langchain, Ollama bindings, Chroma, loaders, etc.).
- **`.gitignore`**  
  Excludes cache, vector DB, logs, histories and `.env`.
- **`Procfile`**  
  `web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🛠 Local Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Chris-JDev/chatbotrag.git
   cd chatbotrag
Create & activate a virtual environment

bash
Copy
Edit
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run Ollama and expose via ngrok

bash
Copy
Edit
# Start Ollama on port 11434
ollama serve --host 0.0.0.0 --port 11434

# In another terminal:
ngrok http 11434
Copy the HTTPS forwarding URL (e.g. https://abcd1234.ngrok-free.app).

Create a .env file

ini
Copy
Edit
OLLAMA_BASE_URL=https://<your-ngrok-url>
Launch the Streamlit app

bash
Copy
Edit
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
Open http://localhost:8501 in your browser.

☁️ Deploy on Render.com
Provision Ollama service

Create a Web Service on Render.

Start command:

css
Copy
Edit
ollama serve --host 0.0.0.0 --port 11434
In Environment settings, add:

ini
Copy
Edit
OLLAMA_ORIGINS=["*"]
Deploy → copy the public URL (e.g. https://my-ollama.onrender.com).

Provision Streamlit service

Create a second Web Service, link to Chris-JDev/chatbotrag.

Build command:

nginx
Copy
Edit
pip install -r requirements.txt
Start command (in Procfile or settings):

nginx
Copy
Edit
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
In Environment settings, add:

ini
Copy
Edit
OLLAMA_BASE_URL=https://my-ollama.onrender.com
Deploy → your app is live.

🔒 Environment Variables
Set in .env locally or via Render dashboard:

ini
Copy
Edit
OLLAMA_BASE_URL=https://<your-ngrok-or-render-url>
