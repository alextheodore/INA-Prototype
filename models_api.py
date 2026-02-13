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

# Versi Lokal
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Inisialisasi Model Lokal (Pastikan Ollama sudah running di laptop)
llm = Ollama(model="deepseek-r1:8b") 

# 2. Inisialisasi Embeddings Multilingual
# Sangat bagus untuk memproses dokumen UMKM berbahasa Indonesia
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)