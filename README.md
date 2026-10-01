# Submission Akhir — Belajar Machine Learning untuk Pemula (BMLP)

Submission akhir kelas **Belajar Machine Learning untuk Pemula (Dicoding)** oleh **Andi Arif Abdillah**.

Proyek ini menggabungkan **unsupervised learning (Clustering)** dan **supervised learning (Klasifikasi)** pada *Bank Transaction Dataset for Fraud Detection*: clustering dipakai untuk membentuk label (`Target`), lalu label tersebut diprediksi dengan model klasifikasi.

## 📚 Belajar dari Nol

Ingin memahami proyek ini luar dalam — dari konsep dasar machine learning sampai alasan di balik setiap sel notebook? Mulai dari **[Panduan Belajar](belajar/README.md)** (9 bab, Bahasa Indonesia, memakai angka asli proyek ini).

| Bab | Topik |
|-----|-------|
| [1](belajar/01-tujuan-proyek.md) | Tujuan proyek |
| [2](belajar/02-fondasi-machine-learning.md) | Fondasi machine learning |
| [3](belajar/03-alat-dan-lingkungan.md) | Alat dan lingkungan |
| [4](belajar/04-mengenal-dataset.md) | Mengenal dataset |
| [5](belajar/05-notebook-clustering.md) | Bedah notebook clustering |
| [6](belajar/06-membaca-hasil-cluster-dengan-jujur.md) | ⚠️ Membaca hasil cluster dengan jujur |
| [7](belajar/07-notebook-klasifikasi.md) | Bedah notebook klasifikasi |
| [8](belajar/08-eksperimen-dan-perbaikan.md) | Eksperimen dan perbaikan |
| [9](belajar/09-glosarium-dan-latihan.md) | Glosarium dan latihan |

## 📁 Struktur Berkas

| Berkas | Keterangan |
|--------|-----------|
| `[Clustering]_Submission_Akhir_BMLP_Andi_Arif_Abdillah.ipynb` | Notebook clustering: EDA, preprocessing, K-Means, interpretasi cluster |
| `[Klasifikasi]_Submission_Akhir_BMLP_Andi_Arif_Abdillah.ipynb` | Notebook klasifikasi: Decision Tree, Random Forest, hyperparameter tuning |
| `model_clustering.h5` | Model K-Means (utama) |
| `PCA_model_clustering.h5` | Model K-Means berbasis PCA |
| `decision_tree_model.h5` | Model Decision Tree |
| `explore_RandomForest_classification.h5` | Model Random Forest |
| `tuning_classification.h5` | Model hasil hyperparameter tuning (GridSearchCV) |
| `data_clustering.csv` | Data hasil clustering (ter-*scale* + kolom `Target`) |
| `data_clustering_inverse.csv` | Data hasil clustering (setelah *inverse* + kolom `Target`) |
| `belajar/` | Panduan belajar + skrip eksperimen |
| `requirements.txt` | Versi library yang dipakai |

## 🔎 Ringkasan Hasil

- **Clustering:** K-Means dengan **k = 2** (Silhouette Score ≈ **0,57**), 1.945 baris setelah pembersihan.
- **Klasifikasi:** Decision Tree, Random Forest, dan Random Forest hasil tuning memprediksi `Target` dengan akurasi & F1-score **1,00**.
- **Level:** Advanced (seluruh kriteria wajib + opsional).

### ⚠️ Catatan Penting tentang Interpretasi

Penyelidikan lanjutan menunjukkan bahwa kedua cluster **ditentukan hampir sepenuhnya oleh kolom `Location`**. Kolom ini di-*LabelEncode* menjadi kode 0–42 (urutan abjad) tetapi tidak diskalakan, sehingga memegang **95,7% variansi** dan mendominasi jarak K-Means. Akibatnya:

- Cluster 0 = kota berkode 0–21 (*Albuquerque … Louisville*), Cluster 1 = kota berkode 22–42 (*Memphis … Washington*).
- Persona "dewasa/profesional vs muda/pelajar" yang tertulis di notebook **tidak didukung data** — proporsi profesi dan kelompok umur di kedua cluster hampir sama.
- Akurasi klasifikasi 100% terjadi karena model cukup menghafal daftar kota; tanpa `Location`, akurasinya turun ke 48% (setara menebak acak).

Bukti lengkap dan koreksinya ada di [Bab 6](belajar/06-membaca-hasil-cluster-dengan-jujur.md); alternatif yang lebih bermakna ada di [Bab 8](belajar/08-eksperimen-dan-perbaikan.md).

## ⚙️ Environment

Python 3.12 · **scikit-learn 1.7.0** · pandas 2.2.3 · numpy 1.26.4 · yellowbrick 1.5 · seaborn · joblib — lihat [`requirements.txt`](requirements.txt).

```bash
python -m venv bmlp_env
bmlp_env\Scripts\activate
pip install -r requirements.txt
```

Dataset dimuat otomatis dari Google Sheets (URL tertanam di notebook), tanpa perlu unduh manual.
