import os
import unicodedata
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# .env dosyasını yükle (GEMINI_API_KEY için)
load_dotenv()

# Gemini API anahtarı
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Embedding modeli (Türkçe destekli, semantik olarak güçlü)
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Kullanılacak LLM modeli
LLM_MODEL = "gemini-2.5-flash"


# ✅ Türkçe karakter farklarını ortadan kaldırmak için normalize fonksiyonu
def normalize_text(text: str) -> str:
    """
    Türkçe karakter farklarını giderir, harfleri normalize eder.
    Örn: 'İstanbul' -> 'istanbul', 'Çalışma' -> 'calisma'
    """
    text = unicodedata.normalize("NFKD", text)
    mapping = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = text.translate(mapping)
    return text.lower().strip()


def build_rag_chain(file_path):
    """
    Belirtilen dosya yolundan veriyi yükler, işler ve RAG zincirini kurar.
    Desteklenen formatlar: .txt ve .pdf
    """

    # 1. Belgeyi uzantısına göre yükle
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Desteklenmeyen dosya formatı. Lütfen .pdf veya .txt yükleyin.")

    documents = loader.load()

    # 2. Belgeleri parçalara ayır
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # ✅ Belgeleri normalize et (Türkçe karakter farkı ortadan kalkar)
    docs = [
        doc.__class__(page_content=normalize_text(doc.page_content), metadata=doc.metadata)
        for doc in docs
    ]

    # 3. Embedding ve FAISS index'i oluştur
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 4. Gemini modelini yükle
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
        max_output_tokens=500
    )

    # 5. Asistanın davranışını belirleyen prompt
    system_prompt = """
    Sen İstanbul Büyükşehir Belediyesi'nin (İBB) faaliyet raporu konusunda uzman bir asistansın.
    Cevaplarını yalnızca rapor içeriğindeki bilgilere dayanarak ver.
    Kullanıcı, İBB'nin yaptığı faaliyetler, projeler, harcamalar veya performans göstergeleri hakkında sorular soracak.
    Cevaplarını Türkçe, açık ve profesyonel bir dille ver.
    Eğer sorunun cevabı metinde yoksa, 'Bu sorunun cevabı sağlanan veri kaynağında bulunmamaktadır.' şeklinde cevapla.
    """

    prompt_template = PromptTemplate(
        template=(
            system_prompt + "\n\n"
            "Bağlam:\n{context}\n\n"
            "Soru: {question}\n\n"
            "Cevap:"
        ),
        input_variables=["context", "question"]
    )

    # 6. RAG zincirini oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain, normalize_text
