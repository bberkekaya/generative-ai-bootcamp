import streamlit as st
import os
import tempfile
from rag_pipeline import build_rag_chain  # normalize_text da dönecek

# 🎨 Sayfa yapılandırması
st.set_page_config(page_title="İBB Faaliyet Raporu RAG", layout="wide")

# 🏙️ Başlık
st.title("📘 Faaliyet Raporu RAG Uygulaması")
st.caption("Google Gemini + Hugging Face + LangChain + Streamlit")

# 🧭 Sidebar
st.sidebar.header("⚙️ Uygulama Bilgileri")
st.sidebar.write("**LLM:** Gemini 2.5 Flash")
st.sidebar.write("**Embedding Model:** paraphrase-multilingual-mpnet-base-v2")
st.sidebar.write("**Vektör Veritabanı:** FAISS (her seferinde yeniden oluşturuluyor)")
st.sidebar.write("---")

# 🚨 API Anahtarı kontrolü
if os.getenv("GEMINI_API_KEY") is None:
    st.error("❌ Lütfen `GEMINI_API_KEY` ortam değişkenini ayarlayın.")
    st.stop()
else:
    st.sidebar.success("✅ API Anahtarı bulundu.")

# 📁 Dosya yükleme
uploaded_file = st.file_uploader(
    "Bir .pdf veya .txt dosyası yükleyin (Örn: 2024_Faaliyet_Raporu.pdf)",
    type=["pdf", "txt"]
)

# 🧩 Model kurulum
if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    st.success(f"✅ Dosya başarıyla yüklendi: {uploaded_file.name}")

    # Zinciri oluştur
    if "qa_chain" not in st.session_state:
        with st.spinner("🔧 Metin bölünüyor, embedding oluşturuluyor ve model hazırlanıyor..."):
            try:
                qa_chain, normalize_text = build_rag_chain(temp_file_path)
                st.session_state.qa_chain = qa_chain
                st.session_state.normalize_text = normalize_text
                st.success("🚀 Model başarıyla hazırlandı! Artık sorular sorabilirsiniz.")
            except Exception as e:
                st.error(f"❌ Model hazırlanırken bir hata oluştu: {e}")
                st.stop()
            finally:
                os.unlink(temp_file_path)

    # 🧠 Soru sorma alanı
    st.markdown("### 🔍 Soru Sorun")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "assistant", "content": "RAG zinciri kuruldu. Lütfen raporunuzla ilgili sorularınızı sorun."}
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Raporla ilgili sorunuzu yazın:")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("💬 Model yanıt üretiyor..."):
                try:
                    qa_chain = st.session_state.qa_chain
                    normalize_text = st.session_state.normalize_text

                    # ✅ Sorguyu normalize et (Türkçe karakter farkı giderilir)
                    normalized_query = normalize_text(user_query)
                    response = qa_chain.invoke(normalized_query)
                    answer = response.get("result", "Cevap bulunamadı.")

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_message = f"❌ Yanıt üretirken bir hata oluştu: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

else:
    if "qa_chain" in st.session_state:
        del st.session_state.qa_chain
    st.info("⬆️ Lütfen .pdf veya .txt dosyasını yükleyin.")
