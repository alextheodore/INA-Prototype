# Import library dan file pendukung
from ina_agent import INAAgent
from models_api import llm # Gunakan models_api karena sebelumnya Anda menggunakan API OpenRouter

def test_agent():
    # Inisialisasi agent menggunakan model yang diimport
    agent = INAAgent(llm)
    
    test_cases = [
        "Saya mau buka usaha bakso, bantu analisis SWOT",
        "Modal saya 20 juta, margin 25%, hitung proyeksi",
        "Usaha fashion cocoknya di platform apa?",
        "Cara daftar KUR yang mudah",
        "Berapa lama proses perizinan UMKM?"
    ]
    
    for test in test_cases:
        print(f"Q: {test}")
        print(f"A: {agent.chat(test)}")
        print("-" * 50)

if __name__ == "__main__":
    test_agent()