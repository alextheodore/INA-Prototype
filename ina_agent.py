from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
# GANTI: Gunakan Window Memory untuk menghemat token (saran Pak Henri)
from langchain.memory import ConversationBufferWindowMemory
import chromadb
# IMPORT: Tambahkan embeddings dari models_api
from models_api import embeddings

class INAAgent:
    def __init__(self, llm):
        self.llm = llm
        # 1. GANTI: Gunakan ConversationBufferWindowMemory dengan k=5
        # Ini akan membatasi ingatan hanya 5 chat terakhir agar hemat token.
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            return_messages=True,
            k=5
        )
        self.setup_tools()
        self.setup_agent()

    def setup_tools(self):
        def analisis_swot(usaha_type: str) -> str:
            return f"Analisis SWOT untuk {usaha_type}: KEKUATAN: Modal kecil. KELEMAHAN: Akses modal terbatas."

        def kalkulator_keuangan(input_str: str) -> str:
            try:
                # Membersihkan input dari karakter non-angka
                clean_input = input_str.replace("Rp", "").replace(".", "").replace(" ", "")
                modal, margin = map(float, clean_input.split(','))
                keuntungan = modal * (margin / 100)
                return f"Modal: Rp {modal:,.0f}, Untung: Rp {keuntungan:,.0f}"
            except:
                return "Format salah. Gunakan: modal,margin (contoh: 10000000,40)"

        def rekomendasi_platform(jenis_usaha: str) -> str:
            return "Rekomendasi: Instagram Ads dan WhatsApp Business untuk pendekatan personal."
        
        def cari_panduan(query: str) -> str:
            # 2. GANTI: Gunakan embeddings lokal agar RAG tetap jalan di model lokal
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection(
                name="umkm_knowledge",
                embedding_function=embeddings
            )
            results = collection.query(query_texts=[query], n_results=1)
            return results['documents'][0][0] if results['documents'] else "Maaf, informasi tidak ditemukan."

        self.tools = [
            Tool(name="Analisis_SWOT", func=analisis_swot, description="Gunakan untuk analisis kekuatan dan kelemahan bisnis"),
            Tool(name="Kalkulator_Keuangan", func=kalkulator_keuangan, description="Hitung untung rugi. Input: modal,margin (tanpa titik/Rp)"),
            Tool(name="Rekomendasi_Platform", func=rekomendasi_platform, description="Saran platform digital marketing"),
            Tool(name="Cari_Panduan_UMKM", func=cari_panduan, description="Cari aturan NIB, perizinan, dan panduan resmi UMKM")
        ]

    def setup_agent(self):
        prefix = """Anda adalah INA, Asisten Pakar Bisnis UMKM Indonesia. 
        Tugas Anda adalah memberikan jawaban yang mendalam, terstruktur, dan solutif.
        
        Saat menjawab pertanyaan bisnis:
        1. Berikan kategori yang jelas (Tradisional, Modern, Online, dll).
        2. Berikan contoh brand nyata yang ada di Indonesia.
        3. Tambahkan tips strategis (Analisis SWOT, Harga, atau Pemasaran).
        4. Gunakan bahasa yang ramah namun profesional.
        """
        
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": prefix
            }
        )

    def chat(self, input_text: str) -> str:
        try:
            # .run() tetap yang paling stabil untuk versi 0.0.350
            response = self.agent_executor.run(input=input_text)
            return response
        except Exception as e:
            return f"Maaf, saya menemui kendala teknis: {str(e)}"