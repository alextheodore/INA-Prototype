from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
 base_url="https://openrouter.ai/api/v1",
 api_key=os.getenv("OPENROUTER_API_KEY"), 
 model="deepseek/deepseek-r1-0528:free",
 model_kwargs={
        "extra_headers": {
            "HTTP-Referer": "http://localhost:8501", 
            "X-Title": "INA Prototype"
        }
    }
)
