import streamlit as st
import sys
import os

# ==========================================================
# FIX IMPORT — memastikan Python memakai modul di /src/
# ==========================================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from search import MedicalSearchEngine
from preprocess import preprocess


#  PAGE CONFIG
st.set_page_config(
    page_title="Medical Search Engine",
    page_icon="🏥",
    layout="wide"
)


#  CUSTOM CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .disease-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        margin-bottom: 1rem;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .matched-term {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin: 0.2rem;
        background: #4CAF50;
        color: white;
        border-radius: 12px;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)


#  LOAD SEARCH ENGINE
@st.cache_resource
def load_engine():
    with st.spinner("🔄 Memuat sistem..."):
        try:
            engine = MedicalSearchEngine(data_dir="data", weighting_scheme="tfidf")
            return engine
        except Exception as e:
            st.error(f"❌ Gagal memuat sistem: {e}")
            st.info("Pastikan folder 'data' berisi file .txt dokumen penyakit")
            return None

engine = load_engine()


# ===========================
# HEADER
# ===========================
st.markdown('<div class="main-title">🏥 Sistem Pencarian Penyakit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Cari penyakit berdasarkan gejala atau kata kunci</div>', unsafe_allow_html=True)

if engine is None:
    st.stop()


# ===========================
# SIDEBAR
# ===========================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    
    model_type = st.radio(
        "Metode Pencarian:",
        ["VSM (TF-IDF)", "Boolean Retrieval"]
    )

    top_k = st.slider("Jumlah Hasil:", 1, 20, 10)
    
    st.markdown("---")
    st.markdown("### 📊 Info Sistem")
    st.markdown("**Model Default:**")
    st.markdown("Vector Space Model (TF-IDF)")
    st.metric("Total Penyakit", len(engine.documents))
    st.metric("Vocabulary", len(engine.vsm.vocab))


# ===========================
# MAIN SEARCH
# ===========================
st.markdown("### 🔍 Pencarian")

query = st.text_input(
    "Masukkan gejala atau kata kunci:",
    placeholder="Contoh: demam AND virus, batuk OR pilek, atau gejala bebas",
    help="Untuk Boolean gunakan operator: AND, OR, NOT"
)

search_btn = st.button("🔍 Cari Penyakit", type="primary", use_container_width=True)


# ===========================
# SEARCH PROCESS
# ===========================
if search_btn and query.strip():

    with st.spinner("🔄 Mencari..."):

        selected_model = "vsm" if model_type == "VSM (TF-IDF)" else "boolean"

        raw_results = engine.search(query, model=selected_model, top_k=top_k)

        # ======================================
        # 🔥 FILTER VSM — hanya tampilkan dokumen
        # yang benar-benar mengandung token query
        # ======================================
        if selected_model == "vsm":
            query_tokens = preprocess(query, use_stemming=True)

            results = []
            for doc_id, score in raw_results:

                # Lewati skor 0 atau sangat kecil
                if score <= 0:
                    continue

                doc = next((d for d in engine.documents if d["doc_id"] == doc_id), None)
                if not doc:
                    continue

                # WAJIB ada kecocokan token
                if any(token in doc["tokens"] for token in query_tokens):
                    results.append((doc_id, score))
        else:
            # Boolean tidak perlu filter
            results = raw_results


    # Jika tidak relevan sama sekali
    if not results:
        st.warning(f"❗ Tidak ada hasil relevan untuk query: **\"{query}\"**")
        st.stop()

    st.success(f"✅ Ditemukan {len(results)} hasil untuk metode **{model_type}**")

    # ===========================
    # DISPLAY RESULTS
    # ===========================
    for rank, (doc_id, score) in enumerate(results, 1):

        doc_data = next((d for d in engine.documents if d['doc_id'] == doc_id), None)
        if not doc_data:
            continue
        
        disease_name = doc_data['disease_name']
        gejala = doc_data['gejala']
        deskripsi = doc_data['deskripsi']

        matched_phrase = query.lower()

        with st.container():
            st.markdown(f"""
            <div class="disease-card">
                <h3 style="color: #2E86AB; margin-bottom: 0.5rem;">#{rank}. {disease_name}</h3>
                <p style="color: #666;"><strong>Skor Kesesuaian:</strong> {score:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**🩺 Gejala:**")
                st.write(gejala)
                
                st.markdown(f"**📋 Deskripsi:**")
                st.write(deskripsi)
            
            with col2:
                st.markdown("**🔎 Kata Kunci / Pola Query:**")
                st.markdown(f'<span class="matched-term">{matched_phrase}</span>', unsafe_allow_html=True)
            
            st.markdown("---")


elif query.strip():
    st.info("👆 Klik tombol **Cari Penyakit** untuk memulai pencarian")


# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; padding: 1rem;">
    <p>Sistem Temu Kembali Informasi - Medical Search Engine</p>
    <p style="font-size: 0.9rem;">UTS STKI • Universitas Dian Nuswantoro • 2025</p>
</div>
""", unsafe_allow_html=True)
