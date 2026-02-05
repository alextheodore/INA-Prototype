import requests
from bs4 import BeautifulSoup
import json

class KnowledgeBaseUMKM:
    def __init__(self):
        self.data = self.load_data()

    def load_data(self):
        """Load data UMKM dari sumber gratis"""
        return {
            "perizinan": {
                "NIB": "Nomor Induk Berusaha - gratis via oss.go.id",
                "IUMK": "Izin Usaha Mikro Kecil - di kelurahan",
                "PIRT": "Untuk usaha makanan - di Dinas Kesehatan",
                "Halal": "Sertifikasi Halal - BPJPH"
            },
            "pendanaan": {
                "KUR": "Kredit Usaha Rakyat - bunga rendah",
                "PNM": "Permodalan Nasional Madani",
                "LPDB": "Lembaga Pengelola Dana Bergulir",
                "Crowdfunding": "Kitabisa.com, Gandengtangan.co.id"
            },
            "platform": {
                "E-commerce": "Tokopedia, Shopee, Bukalapak",
                "Social Commerce": "Instagram, TikTok Shop",
                "Marketplace": "Bukalapak, Blanja.com",
                "Website": "Gratis dengan Carrd.co atau Canva"
            }
        }

    def search(self, query: str):
        """Search dalam knowledge base"""
        results = []
        query = query.lower()
        for category, items in self.data.items():
            for key, value in items.items():
                if query in key.lower() or query in value.lower():
                    results.append(f"{key}: {value}")
        return results[:5] # Return top 5 results