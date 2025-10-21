# generative-ai-bootcamp
🤖 Türkçe RAG Chatbot

Generative AI Bootcamp için hazırlanmış, Türkçe RAG (Retrieval-Augmented Generation) tabanlı chatbot projesi.

📋 Proje Hakkında

Bu proje, Türkçe kaynaklar üzerinden sorulara yanıt verebilen bir AI asistanı oluşturur. Hugging Face veya kendi veri setinizi kullanarak, kullanıcıların sorularına en uygun belgelerden bilgi çekip yanıt üretir.
RAG yaklaşımı sayesinde model, yalnızca önceden eğitildiği veriye değil, aynı zamanda retrieval (bilgi getirme) adımıyla canlı verilere de dayanabilir.

Projede veri kaynağı olarak İBB 2024 yılı faaliyet raporu kullanılmış olup PDF dosyasına https://ibb.istanbul/ibb/faaliyet-raporlari/ web sitesi üzerinden ulaşılabilir.

🛠️ Kullanılan Teknolojiler

LangChain – RAG pipeline framework

Streamlit – Web arayüzü

Sentence Transformers – Embedding modeli

Google Gemini – Text generation modeli

InMemory / FAISS Document Store – Vektör veritabanı

Hugging Face Datasets – Veri seti yönetimi

🚀 Kurulum
1.Gerekli Paketleri Yükleyin

1.1 Sanal ortam oluşturun (opsiyonel)
python -m venv genai-env
source genai-env/bin/activate # macOS/Linux
genai-env\Scripts\activate # Windows

1.2 Paketleri yükleyin
pip install -r requirements.txt

2.API Anahtarlarını Ayarlayın

2.1 Proje kök dizininde .env dosyası oluşturun:
HF_TOKEN=your_huggingface_token_here
GOOGLE_API_KEY=your_google_api_key_here  # Eğer Google AI kullanıyorsanız


Hugging Face Token: Hugging Face Settings

Google API Key: Google AI Studio

3.Uygulamayı Çalıştırın
streamlit run app.py

Tarayıcınızda otomatik olarak açılacaktır (genellikle http://localhost:8501).

📁 Proje Yapısı
.
├── app.py                 # Ana uygulama dosyası (Streamlit)
├── requirements.txt       # Python bağımlılıkları
├── .env                   # API anahtarları (git'e eklenmez)
├── README.md              # Bu dosya
├── data/                  # Opsiyonel: Kullanıcı veri setleri (.pdf veya .txt formatı)
├── rag_pipeline.py        # Proje kodlarının yer aldığı dosya (LangChain, API Key Bağlantısı vb.)
└── venv/                  # Ortam dosyaları

💡 Çalışma Mantığı

Veri Yükleme: Hugging Face veya lokal veri setinden Türkçe belgeler yüklenir

Belge İşleme: Belgeler parçalara ayrılır ve temizlenir

Embedding: Her parça Sentence Transformer ile vektöre dönüştürülür

Vektör Veritabanı: Vektörler InMemory veya FAISS tabanlı store’da saklanır

Sorgulama: Kullanıcının sorusu embedding’e dönüştürülür, en uygun belgeler bulunur

Yanıt Üretimi: Seçilen belgeler kullanılarak LLM modeli yanıt üretir

🎯 Örnek Sorular

"İBB Halk Market Gıda Ürünleri Satış Hizmetleri nelerdir?"

"Tech İstanbul nedir?"

"Yerel Tohum Üretim ve Muhafaza Merkezi nedir?"

⚠️ Önemli Notlar

İlk çalıştırmada veri seti indirilir ve embedding işlemi yapılır; büyük veri setlerinde bu uzun sürebilir

Prompt yazılırken çeşitli harf değişiklikleri LLM'in soruyu yanıtsız bırakmasına neden olabilir

CPU’da embedding işlemi yavaş olabilir; GPU kullanımı önerilir

Streamlit cache mekanizması ile sonraki çalıştırmalar hızlıdır

🐛 Sorun Giderme

ModuleNotFoundError: pip install -r requirements.txt

Veri seti yüklenmiyor (gated dataset hatası): Hugging Face hesabınızla giriş yapın ve HF_TOKEN ekleyin

Embedding işlemi çok yavaş: Küçük veri seti veya GPU kullanın

📝 Lisans

Bu proje eğitim ve araştırma amaçlıdır.


