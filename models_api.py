# from langchain.chat_models import ChatOpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# llm = ChatOpenAI(
#  base_url="https://openrouter.ai/api/v1",
#  api_key=os.getenv("OPENROUTER_API_KEY"), 
#  model="deepseek/deepseek-r1-0528:free",
#  model_kwargs={
#         "extra_headers": {
#             "HTTP-Referer": "http://localhost:8501", 
#             "X-Title": "INA Prototype"
#         }
#     }
# )

# import streamlit as st
# from langchain.chat_models import ChatOpenAI 
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Logika untuk mengambil API Key: 
# # Cek st.secrets dulu (untuk Cloud), jika tidak ada pakai os.getenv (untuk Lokal)
# api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# llm = ChatOpenAI(
#     openai_api_base="https://api.groq.com/openai/v1",
#     openai_api_key=api_key,
#     model_name="qwen/qwen3-32b",
#     temperature=0.6,
#     max_tokens=4096
# )

import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI # Pastikan ini ada di requirements.txt

# Deteksi apakah sedang berjalan di Cloud atau Local
is_cloud = os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud" or st.secrets.get("GROQ_API_KEY")

if is_cloud:
    # --- JIKA DI CLOUD (Pake Groq agar Pak Henri bisa akses) ---
    api_key = st.secrets.get("GROQ_API_KEY")
    llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        model_name="qwen-2.5-32b-instruct"
    )
else:
    # --- JIKA DI LOCALHOST (Pake Ollama agar hemat token) ---
    llm = Ollama(model="llama2:7b")

# Embeddings tetap bisa pake HuggingFace (tapi butuh sentence-transformers di requirements.txt)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)