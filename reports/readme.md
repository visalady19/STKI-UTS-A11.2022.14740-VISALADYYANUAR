Search Penyakit App — Sistem Temu Kembali Informasi Penyakit

Aplikasi ini merupakan implementasi sederhana Sistem Temu Kembali Informasi (Information Retrieval / IR) untuk melakukan pencarian informasi penyakit berbasis teks. Sistem menggunakan dua metode pencarian:

Boolean Retrieval

Vector Space Model (VSM) dengan bobot TF-IDF + Cosine Similarity

Aplikasi dilengkapi preprocessing lengkap (case folding, tokenizing, stopword removal, stemming) dan dijalankan melalui antarmuka sederhana berbasis Streamlit.

Cara Menjalankan Aplikasi

Ikuti langkah-langkah berikut:

1. Clone atau ekstrak proyek

Jika menggunakan Git:
git clone <repository-url>
cd searchpenyakitapp


Jika menggunakan file ZIP, cukup ekstrak kemudian buka foldernya.

2. Install dependencies

Jalankan:
pip install -r requirements.txt

Pastikan library berikut terinstall:
Streamlit
Sastrawi
NumPy
Pandas

3. Jalankan aplikasi Streamlit
streamlit run app/main.py


Berikut asumsi-asumsi yang digunakan dalam proyek:
Dataset penyakit disimpan dalam folder data/ dengan format .txt.
Setiap dokumen mewakili satu jenis penyakit.
Format isi dokumen bebas (tidak wajib mengikuti template khusus).
Bahasa dokumen adalah Bahasa Indonesia.
Preprocessing dilakukan hanya pada level kata, tidak menggunakan NER atau semantic parsing.
Stemming menggunakan library Sastrawi.
Tidak digunakan database eksternal; sistem berjalan sepenuhnya berbasis file.
Query pengguna terdiri dari satu atau lebih kata kunci.
Sistem hanya menangani pencarian berbasis exact term (Boolean) dan term-weighting (VSM), bukan semantic search.
UI menggunakan Streamlit dan berjalan lokal (localhost).