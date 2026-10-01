# Bab 3 — Alat dan Lingkungan

> **Inti bab ini:** mengenal "bengkel" tempat proyek ini dibuat: Jupyter Notebook, library Python yang dipakai, pola kerja scikit-learn, dan kenapa **versi library** bisa membuat proyek berhasil atau gagal total.

---

## 3.1 Jupyter Notebook

Notebook (`.ipynb`) adalah dokumen yang mencampur **teks penjelasan** dan **kode yang bisa dijalankan**.

- **Sel markdown** → teks, judul, gambar.
- **Sel kode** → kode Python yang dijalankan, hasilnya (*output*) muncul di bawahnya.
- **Kernel** → proses Python di belakang layar yang menjalankan sel. Variabel yang dibuat di satu sel **tetap ada** untuk sel berikutnya.

### Kenapa urutan menjalankan sel penting

Kernel mengingat apa pun yang sudah dijalankan, **sesuai urutan kamu menjalankannya**, bukan urutan di layar. Contoh masalah:

1. Kamu menjalankan sel `df.dropna(inplace=True)` → data berkurang.
2. Kamu menjalankannya **lagi** → tidak berubah (sudah tidak ada NaN), aman.
3. Tapi sel outlier `df = df[...]` kalau dijalankan dua kali bisa **membuang lebih banyak baris** dari seharusnya.

Itulah kenapa template Dicoding mewajibkan **Run All** (jalankan semua sel dari atas ke bawah) sebelum submit, supaya hasilnya konsisten dan bisa diulang.

> 💡 Output setiap sel **ikut tersimpan** di dalam file `.ipynb` (formatnya JSON). Itu sebabnya reviewer bisa melihat hasil tanpa menjalankan ulang, dan kenapa file notebook clustering ukurannya sampai ±2 MB (banyak gambar grafik di dalamnya).

---

## 3.2 Library yang Dipakai

| Library | Fungsi umum | Dipakai untuk apa di proyek ini |
|---|---|---|
| **pandas** | Mengolah data tabel (`DataFrame`) | Memuat CSV, `head()`, `info()`, `dropna()`, `groupby()`, `get_dummies()` |
| **numpy** | Perhitungan numerik cepat | Dipakai di balik layar oleh pandas & scikit-learn |
| **matplotlib** | Menggambar grafik dasar | Kanvas grafik, judul, label sumbu |
| **seaborn** | Grafik statistik yang lebih cantik | Heatmap korelasi, histogram, boxplot, scatterplot |
| **scikit-learn** | Library machine learning utama | `LabelEncoder`, `StandardScaler`, `KMeans`, `PCA`, `silhouette_score`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`, metrik |
| **yellowbrick** | Visualisasi untuk model scikit-learn | `KElbowVisualizer` untuk memilih jumlah cluster |
| **joblib** | Menyimpan/memuat objek Python ke file | Menyimpan model ke file `.h5` |

---

## 3.3 Pola Kerja scikit-learn: `fit`, `transform`, `predict`

Hampir semua objek di scikit-learn mengikuti pola yang sama. Kalau kamu paham pola ini, kamu paham 80% cara memakai scikit-learn.

| Method | Arti | Contoh di proyek ini |
|---|---|---|
| `fit(X)` | **Belajar** dari data (menghitung parameter) | `StandardScaler` menghitung rata-rata & standar deviasi; `KMeans` mencari pusat cluster |
| `transform(X)` | **Menerapkan** apa yang sudah dipelajari | `StandardScaler` mengubah angka jadi skala baru |
| `fit_transform(X)` | `fit` lalu `transform` sekaligus | `scaler.fit_transform(df[numerical_cols])` |
| `inverse_transform(X)` | **Membalikkan** transformasi | Mengembalikan angka ter-*scale* ke nilai aslinya |
| `fit(X, y)` | Belajar dari fitur **dan** label (supervised) | `decision_tree_model.fit(X_train, y_train)` |
| `predict(X)` | **Menebak** untuk data baru | `decision_tree_model.predict(X_test)` |

Contoh nyata dari proyek ini:

```python
scaler = StandardScaler()
scaler.fit(df[numerical_cols])          # belajar: mean = 256,84 ; std = 218,31 (TransactionAmount)
hasil  = scaler.transform(df[numerical_cols])   # terapkan: (x - 256,84) / 218,31
asli   = scaler.inverse_transform(hasil)        # balik: hasil * 218,31 + 256,84
```

> ⚠️ **Aturan emas:** objek yang sudah di-`fit` **menyimpan apa yang dipelajarinya**. Itulah kenapa variabel `scaler` dan `encoders` harus tetap ada sampai tahap *inverse* di akhir notebook — kalau hilang, kita tidak bisa mengembalikan data ke nilai aslinya.

---

## 3.4 Kenapa Versi Library Itu Penting (Pengalaman Nyata Proyek Ini)

Dicoding menyarankan **scikit-learn 1.7.0**. Saat mengerjakan proyek ini, komputer awalnya memakai versi yang **lebih baru**, dan itu menimbulkan **empat masalah nyata**:

### Masalah 1 — pandas 3.0 mengubah tipe data teks
Di pandas 2.x, kolom teks bertipe `object`. Di pandas 3.0, kolom teks bertipe **`str`**. Akibatnya:

```python
df.select_dtypes(include=['object']).columns   # pandas 3.0 → KOSONG!
```

Kolom kategorikal tidak terdeteksi → tidak ada yang di-encode → K-Means gagal karena masih ada teks. **Solusi:** pakai pandas 2.2.3.

### Masalah 2 — yellowbrick tidak mengenali KMeans versi baru
Dengan scikit-learn 1.8, `KElbowVisualizer` menolak model:

```
YellowbrickTypeError: The supplied model is not a clustering estimator
```

yellowbrick 1.5 memeriksa "jenis" model dengan cara lama yang sudah berubah di scikit-learn terbaru. **Solusi:** pakai scikit-learn 1.7.0.

### Masalah 3 — yellowbrick butuh modul yang sudah dihapus Python
yellowbrick 1.5 memakai `distutils`, modul yang **dihapus dari Python 3.12**:

```
ModuleNotFoundError: No module named 'distutils'
```

**Solusi:** install `setuptools`, yang menyediakan pengganti `distutils`.

### Masalah 4 — model yang disimpan terikat versi
Model yang disimpan dengan scikit-learn versi X sebaiknya dimuat dengan versi X juga. Reviewer Dicoding memuat model secara otomatis dengan versi yang mereka pakai, jadi menyamakan versi mencegah error atau peringatan `InconsistentVersionWarning`.

> 📌 **Pelajaran:** "kodenya benar tapi tetap error" sering kali bukan salah kode, melainkan **salah versi**. Selalu catat versi library yang dipakai.

---

## 3.5 Virtual Environment

**Virtual environment** adalah "ruang Python terpisah" dengan versi library-nya sendiri. Proyek ini memakai environment bernama `bmlp_env` supaya versinya tidak bentrok dengan proyek lain di komputer yang sama.

Versi yang dipakai proyek ini tercatat di [`requirements.txt`](../requirements.txt):

| Paket | Versi |
|---|---|
| scikit-learn | 1.7.0 |
| pandas | 2.2.3 |
| numpy | 1.26.4 |
| yellowbrick | 1.5 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| joblib | 1.5.3 |
| setuptools | 80.10.2 |

### Cara membuat ulang environment (Windows)

```bash
python -m venv bmlp_env
bmlp_env\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name bmlp_env --display-name "Python (BMLP sklearn 1.7)"
```

Setelah itu, buka notebook dan pilih kernel **"Python (BMLP sklearn 1.7)"**.

---

## 3.6 Kebenaran tentang File `.h5`

Template kelas menyimpan model seperti ini:

```python
joblib.dump(model, "model_clustering.h5")
```

Banyak orang mengira `.h5` berarti format **HDF5** (format yang biasa dipakai Keras/TensorFlow). Kenyataannya:

- `joblib.dump` **selalu** menyimpan dengan format joblib (berbasis *pickle*), **apa pun akhiran nama filenya**.
- Akhiran `.h5` di proyek ini hanya **konvensi penamaan** dari kelas.
- Cara memuatnya: `joblib.load("model_clustering.h5")`.

### Contoh memakai model yang sudah disimpan

```python
import joblib
import pandas as pd

kmeans = joblib.load("model_clustering.h5")
data   = pd.read_csv("data_clustering.csv")

X = data.drop(columns="Target")            # 10 fitur, urutan sama seperti saat latihan
cocok = (kmeans.predict(X) == data["Target"]).mean()
print(cocok)   # 1.0 → model memberi label yang sama persis dengan kolom Target
```

> ⚠️ **Keamanan:** file *pickle* bisa berisi kode yang ikut dijalankan saat dimuat. **Jangan pernah** `joblib.load` file model dari sumber yang tidak kamu percayai.

---

**← Sebelumnya:** [Bab 2 — Fondasi Machine Learning](02-fondasi-machine-learning.md) · **Selanjutnya:** [Bab 4 — Mengenal Dataset →](04-mengenal-dataset.md)
