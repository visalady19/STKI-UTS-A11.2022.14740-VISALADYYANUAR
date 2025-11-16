import re
from pathlib import Path
from typing import List, Dict
from collections import Counter

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    STEMMER_AVAILABLE = True
except ImportError:
    stemmer = None
    STEMMER_AVAILABLE = False
    print("Warning: Sastrawi not installed. Stemming will be skipped.")

# Stopwords untuk domain medis (lebih selektif)
MEDICAL_STOPWORDS = set([
    "yang", "dan", "atau", "di", "ke", "dari", "pada", "untuk", "dengan",
    "jika", "bahwa", "adalah", "itu", "ini", "dalam", "sebagai", "oleh",
    "akan", "telah", "sudah", "dapat", "bisa", "ada", "tidak", "ia",
    "kita", "kami", "saya", "anda", "nya", "se", "ter", "paling",
    "sangat", "lebih", "juga", "masih", "saat", "hingga", "seperti"
])

# Term medis penting - JANGAN di-stem untuk menjaga akurasi
PRESERVE_MEDICAL_TERMS = set([
    "demam", "batuk", "flu", "diare", "mual", "muntah", "pusing", "pucat",
    "nyeri", "sakit", "lemas", "sesak", "gatal", "ruam", "bengkak",
    "nafas", "kepala", "perut", "dada", "tenggorokan", "mata", "kulit",
    "darah", "merah", "tinggi", "rendah", "kering", "basah", "akut", "kronis"
])

def clean_text(text: str) -> str:
    """
    Membersihkan teks:
    - Lowercase
    - Hapus angka dan karakter spesial
    - Normalisasi whitespace
    """
    text = text.lower()
    # Hapus "gejala:", "deskripsi:" untuk fokus ke konten
    text = re.sub(r'(gejala|deskripsi)\s*:', '', text)
    # Hapus angka dan karakter non-alfabet
    text = re.sub(r"[^a-z\s]", " ", text)
    # Normalisasi whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """Memecah teks menjadi token/kata"""
    return text.split()

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Menghapus stopwords dan token terlalu pendek"""
    return [t for t in tokens if t not in MEDICAL_STOPWORDS and len(t) > 2]

def stem_tokens_medical(tokens: List[str]) -> List[str]:
    """
    Stemming khusus untuk domain medis
    Preserve term medis penting untuk akurasi diagnosis
    """
    if not STEMMER_AVAILABLE or stemmer is None:
        return tokens
    
    result = []
    for token in tokens:
        if token in PRESERVE_MEDICAL_TERMS:
            result.append(token)  # Jangan di-stem
        else:
            result.append(stemmer.stem(token))
    return result

def preprocess(text: str, use_stemming: bool = True) -> List[str]:
    """
    Pipeline preprocessing lengkap untuk dokumen medis
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem_tokens_medical(tokens)
    return tokens

def extract_disease_info(text: str) -> Dict[str, str]:
    """
    Ekstrak gejala, deskripsi, dan rekomendasi dari dokumen penyakit
    Format: "Gejala: ... Deskripsi: ... Rekomendasi: ..."
    """
    text_lower = text.lower()
    
    # Extract gejala
    gejala = ""
    if "gejala:" in text_lower:
        parts = text_lower.split("gejala:", 1)
        if len(parts) > 1:
            gejala_part = parts[1].split("deskripsi:", 1)[0]
            gejala = gejala_part.strip()
    
    # Extract deskripsi
    deskripsi = ""
    if "deskripsi:" in text_lower:
        parts = text_lower.split("deskripsi:", 1)
        if len(parts) > 1:
            deskripsi_part = parts[1].split("rekomendasi:", 1)[0]
            deskripsi = deskripsi_part.strip()
    
    # Extract rekomendasi
    rekomendasi = ""
    if "rekomendasi:" in text_lower:
        parts = text_lower.split("rekomendasi:", 1)
        if len(parts) > 1:
            rekomendasi = parts[1].strip()
    
    return {
        'gejala': gejala,
        'deskripsi': deskripsi,
        'rekomendasi': rekomendasi,
        'full_text': text
    }

def preprocess_disease_document(doc_path: str, use_stemming: bool = True) -> Dict:
    """
    Memproses dokumen penyakit dengan ekstraksi struktur
    """
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Ekstrak nama penyakit dari nama file
        disease_name = Path(doc_path).stem
        
        # Ekstrak gejala dan deskripsi
        info = extract_disease_info(raw_text)
        
        # Preprocess semua teks
        all_tokens = preprocess(raw_text, use_stemming)
        
        # Preprocess gejala saja (untuk filtering)
        gejala_tokens = preprocess(info['gejala'], use_stemming) if info['gejala'] else []
        
        return {
            'doc_id': disease_name,
            'disease_name': disease_name.replace('_', ' '),
            'path': doc_path,
            'raw_text': raw_text,
            'gejala': info['gejala'],
            'deskripsi': info['deskripsi'],
            'tokens': all_tokens,
            'gejala_tokens': gejala_tokens,
            'token_count': len(all_tokens),
            'unique_tokens': len(set(all_tokens))
        }
    except Exception as e:
        print(f"Error processing {doc_path}: {e}")
        return None

def preprocess_corpus(data_dir: str, use_stemming: bool = True) -> List[Dict]:
    """
    Memproses seluruh corpus dokumen penyakit
    """
    data_path = Path(data_dir)
    documents = []
    
    # Ambil semua file .txt
    txt_files = sorted(data_path.glob("*.txt"))
    
    print(f"Found {len(txt_files)} disease documents in {data_dir}")
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"Processing [{i}/{len(txt_files)}]: {txt_file.name}")
        doc_data = preprocess_disease_document(str(txt_file), use_stemming)
        if doc_data:
            documents.append(doc_data)
    
    print(f"\nSuccessfully processed {len(documents)} documents")
    return documents

def show_disease_info(doc: Dict):
    """Menampilkan info penyakit dengan format rapi"""
    print(f"\n{'='*80}")
    print(f"PENYAKIT: {doc['disease_name']}")
    print(f"{'='*80}")
    print(f"Gejala: {doc['gejala']}")
    print(f"Deskripsi: {doc['deskripsi']}")
    print(f"\nTokens (preprocessed): {doc['tokens'][:15]}...")
    print(f"Gejala tokens: {doc['gejala_tokens']}")
    print(f"Total tokens: {doc['token_count']}, Unique: {doc['unique_tokens']}")
    print(f"{'='*80}\n")

def get_symptom_frequency(documents: List[Dict]) -> Dict[str, int]:
    """Menghitung frekuensi gejala di seluruh corpus"""
    symptom_counter = Counter()
    for doc in documents:
        symptom_counter.update(doc['gejala_tokens'])
    return dict(symptom_counter)

def get_corpus_stats(documents: List[Dict]):
    """Menampilkan statistik corpus penyakit"""
    total_docs = len(documents)
    total_tokens = sum(doc['token_count'] for doc in documents)
    avg_tokens = total_tokens / total_docs if total_docs > 0 else 0
    
    all_unique = set()
    for doc in documents:
        all_unique.update(doc['tokens'])
    
    # Hitung gejala paling umum
    symptom_freq = get_symptom_frequency(documents)
    common_symptoms = sorted(symptom_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n{'='*80}")
    print("MEDICAL CORPUS STATISTICS")
    print(f"{'='*80}")
    print(f"Total diseases: {total_docs}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per disease: {avg_tokens:.2f}")
    print(f"Vocabulary size: {len(all_unique):,}")
    
    print(f"\nTop 10 Most Common Symptoms:")
    for symptom, count in common_symptoms:
        print(f"  {symptom}: appears in {count} disease(s)")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Test dengan sample data
    sample_text = "Gejala: pucat, lemas, pusing. Deskripsi: Kekurangan sel darah merah."
    
    print("Sample Text:", sample_text)
    print("\nExtracted Info:")
    info = extract_disease_info(sample_text)
    print(f"  Gejala: {info['gejala']}")
    print(f"  Deskripsi: {info['deskripsi']}")
    
    print("\nPreprocessed Tokens:")
    tokens = preprocess(sample_text)
    print(f"  {tokens}")
    
    # Test corpus processing (jika folder data ada)
    try:
        docs = preprocess_corpus("data", use_stemming=True)
        get_corpus_stats(docs)
        
        # Show sample disease
        if docs:
            show_disease_info(docs[0])
    except FileNotFoundError:
        print("\nFolder 'data' not found. Skipping corpus processing.")