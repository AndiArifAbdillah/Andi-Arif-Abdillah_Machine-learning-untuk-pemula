# Submission Akhir — Belajar Machine Learning untuk Pemula (BMLP)

Submission akhir kelas **Belajar Machine Learning untuk Pemula (Dicoding)** oleh **Andi Arif Abdillah**.

Proyek ini menggabungkan **unsupervised learning (Clustering)** dan **supervised learning (Klasifikasi)** pada *Bank Transaction Dataset for Fraud Detection*: clustering dipakai untuk membentuk label (`Target`), lalu label tersebut diprediksi dengan model klasifikasi.

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

## 🔎 Ringkasan Hasil

- **Clustering:** K-Means dengan **k = 2** (Elbow Method + **Silhouette Score ≈ 0.57**).
- **Interpretasi:** Cluster 0 ≈ nasabah dewasa/profesional, Cluster 1 ≈ nasabah muda/pelajar (pembeda utama pada fitur kategorikal).
- **Klasifikasi:** memprediksi label cluster (`Target`) dengan akurasi & F1-score = **1.00**.
- **Level:** Advanced (seluruh kriteria wajib + opsional).

## ⚙️ Environment

Python 3.12 · **scikit-learn 1.7.0** · pandas 2.2.3 · numpy 1.26.4 · yellowbrick 1.5 · seaborn · joblib

Dataset dimuat otomatis dari Google Sheets (URL tertanam di notebook), tanpa perlu unduh manual.
