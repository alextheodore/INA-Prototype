from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
import chromadb

class INAAgent:
    def __init__(self, llm):
        self.llm = llm
        # 1. Tambahkan return_messages=True dan pastikan memory_key sesuai standar agent
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.setup_tools()
        self.setup_agent()

    def setup_tools(self):
        def analisis_swot(usaha_type: str) -> str:
            return f"Analisis SWOT untuk {usaha_type}: KEKUATAN: Modal kecil. KELEMAHAN: Akses modal terbatas."

        def kalkulator_keuangan(input_str: str) -> str:
            try:
                modal, margin = map(float, input_str.split(','))
                keuntungan = modal * (margin / 100)
                return f"Modal: Rp {modal:,.0f}, Untung: Rp {keuntungan:,.0f}"
            except:
                return "Format salah. Gunakan: modal,margin"

        def rekomendasi_platform(jenis_usaha: str) -> str:
            return "Rekomendasi: Instagram dan WhatsApp Business."
        
        def cari_panduan(query: str) -> str:
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection(name="umkm_knowledge")
            results = collection.query(query_texts=[query], n_results=1)
            return results['documents'][0][0] if results['documents'] else "Maaf, informasi tidak ditemukan."

        self.tools = [
            Tool(name="Analisis_SWOT", func=analisis_swot, description="Analisis bisnis"),
            Tool(name="Kalkulator_Keuangan", func=kalkulator_keuangan, description="Hitung keuangan, input: modal"),
            Tool(name="Rekomendasi_Platform", func=rekomendasi_platform, description="Saran platform"),
            Tool(name="Cari_Panduan_UMKM", func=cari_panduan, description="Cari aturan perizinan, NIB, dan panduan UMKM")
        ]

    def setup_agent(self):
        # 2. JANGAN pakai hub.pull atau AgentExecutor manual.
        # initialize_agent versi 0.0.350 sudah merangkap semuanya.
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )

    def chat(self, input_text: str) -> str:
        try:
            # 3. Gunakan .run() untuk versi lama agar lebih stabil
            response = self.agent_executor.run(input=input_text)
            return response
        except Exception as e:
            # Jika error parsing terjadi, kita tangkap pesannya
            return f"Maaf, saya menemui kendala: {str(e)}"