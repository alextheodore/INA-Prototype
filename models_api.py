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

import streamlit as st
from langchain.chat_models import ChatOpenAI # Gunakan library yang lebih baru jika sudah update
import os
from dotenv import load_dotenv

load_dotenv()

# Logika untuk mengambil API Key: 
# Cek st.secrets dulu (untuk Cloud), jika tidak ada pakai os.getenv (untuk Lokal)
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)