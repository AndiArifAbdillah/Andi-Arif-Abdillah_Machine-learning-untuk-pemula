# Bab 9 — Glosarium dan Latihan

> **Inti bab ini:** kamus istilah untuk dibuka kapan saja, soal latihan bertingkat untuk menguji pemahaman, dan cara menjelaskan proyek ini dalam satu menit.

---

## 9.1 Glosarium

| Istilah | Arti singkat | Di proyek ini |
|---|---|---|
| **Akurasi** (*accuracy*) | Persentase tebakan yang benar | 100% di ketiga model klasifikasi |
| **Binning** | Mengubah angka kontinu menjadi kelompok | `CustomerAge` → Muda / Dewasa / Senior |
| **Bootstrap** | Mengambil sampel acak *dengan pengembalian* | Cara Random Forest membuat data untuk tiap pohon |
| **Centroid** | Titik pusat sebuah cluster | Location 10,73 vs 32,04 |
| **Clustering** | Mengelompokkan data yang mirip tanpa label | K-Means dengan k = 2 |
| **Confusion matrix** | Tabel tebakan benar/salah per kelas | 196 / 0 / 0 / 193 |
| **Cross-validation** | Menguji model berulang kali dengan bagian data yang bergiliran | `cv=5` di GridSearchCV |
| **DataFrame** | Tabel data di pandas | Variabel `df` |
| **Decision Tree** | Model berupa rangkaian pertanyaan ya/tidak | Kedalaman 21, menanyakan 21 kota |
| **EDA** | *Exploratory Data Analysis*, mengenali data sebelum memodelkan | `head`, `info`, `describe`, heatmap, histogram |
| **Elbow method** | Cara memilih k dengan mencari "siku" grafik | `KElbowVisualizer` |
| **Encoding** | Mengubah kategori menjadi angka | LabelEncoder, one-hot |
| **F1-Score** | Rata-rata harmonik precision dan recall | 1,00 |
| **Fitur** (*feature*, X) | Kolom masukan model | 10 fitur di clustering, 55 di klasifikasi |
| **`fit` / `transform` / `predict`** | Belajar / menerapkan / menebak | Pola dasar scikit-learn |
| **Generalisasi** | Kemampuan model bekerja di data baru | Diuji dengan data uji 20% |
| **Gini impurity** | Ukuran ketidakmurnian kelompok | 0,5 di akar pohon |
| **GridSearchCV** | Mencoba semua kombinasi hyperparameter + cross-validation | 12 kombinasi × 5 fold |
| **Hyperparameter** | Pengaturan model yang ditentukan sebelum melatih | `n_estimators`, `max_depth` |
| **Imputasi** | Mengisi data kosong dengan nilai perkiraan | Tidak dipakai (diganti `dropna`) |
| **Inertia** | Jumlah kuadrat jarak titik ke pusat cluster-nya | 87.891 pada k = 2 |
| **Inverse transform** | Membalikkan transformasi ke nilai asli | `scaler.inverse_transform` |
| **IQR** | Q3 − Q1, lebar 50% data bagian tengah | Dipakai untuk membuang outlier |
| **joblib** | Library untuk menyimpan objek Python | Membuat file `.h5` |
| **K-Means** | Algoritma clustering berbasis pusat & jarak | Model utama clustering |
| **k-means++** | Cara memilih pusat awal yang saling berjauhan | Bawaan `KMeans` |
| **Kategorikal nominal** | Kategori tanpa urutan | Location, Channel |
| **Kategorikal ordinal** | Kategori dengan urutan | Muda < Dewasa < Senior |
| **Kernel** (Jupyter) | Proses Python yang menjalankan sel notebook | `bmlp_env` |
| **Korelasi** | Kekuatan hubungan linear dua variabel (−1 s.d. 1) | Umur–saldo 0,32 |
| **Label / Target** (y) | Kolom yang ingin ditebak | Kolom `Target` hasil K-Means |
| **LabelEncoder** | Memberi nomor 0, 1, 2 … sesuai abjad | Penyebab masalah `Location` |
| **Missing value / NaN** | Sel data yang kosong | 381 baris punya minimal satu |
| **Modus** | Nilai yang paling sering muncul | Charlotte vs Tucson |
| **One-hot encoding** | Satu kolom 0/1 per kategori | `pd.get_dummies` di klasifikasi |
| **Outlier** | Nilai ekstrem yang jauh dari mayoritas | 190 baris dibuang |
| **Overfitting** | Model menghafal data latih, gagal di data baru | *Bukan* penyebab akurasi 100% di sini |
| **PCA** | Meringkas banyak fitur jadi sedikit arah utama | PCA1 = 95,7% variansi = Location |
| **Pickle** | Format penyimpanan objek Python | Isi sebenarnya file `.h5` |
| **Pipeline / ColumnTransformer** | Merangkai langkah pra-pemrosesan & model dalam satu objek | Disarankan di Bab 8 |
| **Precision** | Dari yang ditebak positif, berapa yang benar | 1,00 |
| **Random Forest** | Banyak pohon keputusan + voting | 100 pohon |
| **`random_state`** | Pengunci keacakan agar hasil bisa diulang | 42 |
| **Recall** | Dari yang sebenarnya positif, berapa yang ditemukan | 1,00 |
| **Silhouette Score** | Ukuran kerapatan & keterpisahan cluster (−1 s.d. 1) | 0,572 |
| **StandardScaler / z-score** | Mengubah data jadi rata-rata 0, std 1 | Hanya untuk 5 kolom numerik |
| **Stratify** | Menjaga proporsi kelas saat membagi data | `stratify=y` |
| **Supervised learning** | Belajar dari data berlabel | Notebook Klasifikasi |
| **Support** | Jumlah data sebenarnya per kelas di laporan | 196 dan 193 |
| **Unsupervised learning** | Belajar dari data tanpa label | Notebook Clustering |
| **Variansi** | Ukuran sebaran data (std²) | Location memegang 95,7% |
| **Virtual environment** | Ruang Python terpisah dengan versi library sendiri | `bmlp_env` |

---

## 9.2 Latihan

Coba jawab dulu sebelum membuka jawabannya.

### Level 1 — Konsep Dasar

**1.** Apa beda *supervised* dan *unsupervised learning*? Sebutkan bagian mana dari proyek ini yang termasuk masing-masing.

<details><summary>Jawaban</summary>

*Supervised* belajar dari data yang punya label (jawaban), *unsupervised* mencari pola tanpa label. Notebook Clustering (K-Means, PCA) = unsupervised. Notebook Klasifikasi (Decision Tree, Random Forest) = supervised, dengan label `Target` yang dibuat oleh K-Means.
</details>

**2.** Kenapa algoritma berbasis jarak seperti K-Means butuh penskalaan fitur?

<details><summary>Jawaban</summary>

Karena jarak Euclidean menjumlahkan selisih setiap fitur. Fitur dengan rentang angka besar (misalnya saldo ribuan) akan mendominasi fitur dengan rentang kecil (misalnya umur puluhan), meskipun belum tentu lebih penting. Penskalaan membuat setiap fitur berkontribusi setara.
</details>

**3.** Apa beda kategori *nominal* dan *ordinal*? Beri satu contoh masing-masing dari dataset ini.

<details><summary>Jawaban</summary>

Nominal tidak punya urutan (`Location`, `Channel`, `CustomerOccupation`). Ordinal punya urutan (`CustomerAge_Group`: Muda < Dewasa < Senior).
</details>

**4.** Kenapa data harus dibagi menjadi data latih dan data uji?

<details><summary>Jawaban</summary>

Supaya kita bisa mengukur kemampuan model pada data yang **belum pernah dilihat** (generalisasi). Menguji dengan data latih seperti memberi ujian dengan soal latihan yang sama — nilai bagus tidak membuktikan pemahaman.
</details>

**5.** Apa yang diukur oleh Silhouette Score, dan apa yang **tidak** bisa diukurnya?

<details><summary>Jawaban</summary>

Silhouette mengukur seberapa rapat titik dengan cluster-nya sendiri dibanding dengan cluster terdekat lainnya (bentuk geometris). Ia **tidak** bisa mengukur apakah pemisahnya bermakna bagi manusia atau bisnis.
</details>

### Level 2 — Memahami Notebook

**6.** Kenapa `duplicated().sum()` di notebook menghasilkan 21, padahal di dataset Kaggle hasilnya 0?

<details><summary>Jawaban</summary>

Karena notebook memakai versi Google Sheets yang **sengaja** disisipi 21 baris duplikat (dan data kosong) oleh Dicoding, supaya langkah pembersihan benar-benar bekerja dan output bisa dicocokkan dengan output yang diharapkan.
</details>

**7.** Setelah langkah pembuangan outlier, kenapa semua nilai `LoginAttempts` menjadi 1?

<details><summary>Jawaban</summary>

Karena lebih dari 75% nilainya 1, sehingga Q1 = Q3 = 1 dan IQR = 0. Batas bawah dan atas sama-sama 1, sehingga semua baris dengan nilai selain 1 (97 baris) dibuang.
</details>

**8.** Kenapa variabel `scaler` dan `encoders` harus tetap disimpan sampai akhir notebook?

<details><summary>Jawaban</summary>

Karena keduanya menyimpan apa yang mereka pelajari (rata-rata/std dan daftar kategori). Tanpa itu, kita tidak bisa melakukan `inverse_transform` untuk mengembalikan data ke nilai dan teks aslinya.
</details>

**9.** Hitung z-score untuk `CustomerAge` = 70 tahun, jika μ = 44,69 dan σ = 17,74.

<details><summary>Jawaban</summary>

z = (70 − 44,69) / 17,74 = 25,31 / 17,74 ≈ **1,43**. Artinya sekitar 1,4 standar deviasi di atas rata-rata. (Di `data_clustering.csv`, baris pertama memang bernilai 1,427.)
</details>

**10.** Apa fungsi `stratify=y` di `train_test_split`?

<details><summary>Jawaban</summary>

Menjaga agar proporsi tiap kelas di data latih dan data uji sama dengan proporsi di seluruh data (di sini sekitar 50:50), sehingga evaluasi lebih adil.
</details>

**11.** GridSearchCV mencoba berapa kali pelatihan model di proyek ini? Tunjukkan perhitungannya.

<details><summary>Jawaban</summary>

2 nilai `n_estimators` × 3 nilai `max_depth` × 2 nilai `min_samples_split` = 12 kombinasi. Masing-masing diuji dengan 5-fold cross-validation → 12 × 5 = **60** pelatihan, ditambah **1** pelatihan ulang dengan kombinasi terbaik pada seluruh data latih.
</details>

### Level 3 — Berpikir Kritis

**12.** Rata-rata semua fitur numerik di kedua cluster hampir sama, tapi Silhouette Score-nya 0,572. Apa yang seharusnya kamu curigai, dan bagaimana membuktikannya?

<details><summary>Jawaban</summary>

Curigai bahwa ada fitur **yang tidak ditampilkan** di tabel agregasi yang memisahkan cluster. Buktikan dengan melihat standar deviasi semua fitur, koordinat centroid untuk **semua** fitur, dan bobot komponen PCA. Ketiganya menunjuk ke `Location`.
</details>

**13.** Kenapa akurasi 100% di Notebook Klasifikasi **bukan** tanda model yang hebat, tapi juga **bukan** overfitting?

<details><summary>Jawaban</summary>

Bukan hebat karena tugasnya melingkar dan terlalu mudah: `Target` ditentukan sepenuhnya oleh `Location`, dan `Location` diberikan ke model. Bukan overfitting karena akurasi di data **uji** juga 100% — overfitting ditandai dengan akurasi uji yang turun.
</details>

**14.** Modus profesi di Cluster 0 adalah "Doctor". Kenapa itu tidak cukup untuk menyebut Cluster 0 sebagai "cluster dokter"?

<details><summary>Jawaban</summary>

Modus hanya menunjukkan nilai yang paling sering, bukan seberapa dominan. Doctor hanya 27,6% di Cluster 0 (vs 23,9% di Cluster 1), dan keempat profesi sama-sama sekitar 25%. Selisih sekecil itu adalah fluktuasi acak.
</details>

**15.** Dataset ini berjudul *Fraud Detection*. Langkah mana di notebook yang paling bertentangan dengan tujuan mendeteksi penipuan? Jelaskan.

<details><summary>Jawaban</summary>

Pembuangan outlier. Langkah itu membuang 93 transaksi bernominal sangat besar dan 97 transaksi dengan percobaan login berulang — justru pola yang paling mencurigakan dalam konteks penipuan.
</details>

**16.** Eksperimen A di Bab 8 punya silhouette 0,244, jauh di bawah 0,572. Kenapa hasil A tetap lebih baik?

<details><summary>Jawaban</summary>

Karena cluster-nya punya **makna** yang bisa dijelaskan dan didukung angka (transaksi besar / senior bersaldo tinggi / muda bersaldo rendah), sedangkan cluster di notebook hanya membagi kota berdasarkan abjad. Silhouette tinggi tidak menjamin makna.
</details>

### Level 4 — Praktik

**17.** Jalankan `belajar/eksperimen/verifikasi_dan_alternatif.py`. Cocokkan setiap angka di output dengan tabel di Bab 6 dan Bab 8.

**18.** Ubah eksperimen A untuk k = 4. Bagaimana profil cluster-nya? Apakah keempat cluster masih bisa diberi nama yang masuk akal?

<details><summary>Petunjuk</summary>

Ganti `n_clusters=3` menjadi `n_clusters=4` pada baris `lab3 = KMeans(...)`. Bandingkan rata-rata setiap fitur antar cluster, lalu coba beri nama setiap cluster dalam satu kalimat. Kalau ada dua cluster yang sulit dibedakan, itu tanda k terlalu besar.
</details>

**19.** Dengan data **mentah** (sebelum outlier dibuang), latih `IsolationForest` dan lihat apakah transaksi yang dianggap anomali cenderung punya `LoginAttempts` > 1 atau nominal besar.

<details><summary>Petunjuk</summary>

```python
import pandas as pd
from sklearn.ensemble import IsolationForest

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?output=csv"
df_mentah = pd.read_csv(url)          # data mentah, sebelum pembersihan apa pun

fitur = ["TransactionAmount", "TransactionDuration", "LoginAttempts", "AccountBalance"]
data = df_mentah.dropna(subset=fitur).copy()
iso = IsolationForest(contamination=0.05, random_state=42).fit(data[fitur])
data["anomali"] = iso.predict(data[fitur]) == -1
print(data.groupby("anomali")[fitur].mean())
```

`contamination=0.05` berarti kita memperkirakan sekitar 5% data adalah anomali. `.copy()` mencegah peringatan pandas saat menambah kolom baru. Bandingkan baris `True` (anomali) dan `False`: fitur mana yang paling berbeda?
</details>

---

## 9.3 Menjelaskan Proyek Ini dalam Satu Menit

Kalau ditanya reviewer, dosen, atau pewawancara, kira-kira begini:

> *"Proyek ini menggabungkan unsupervised dan supervised learning pada data transaksi bank tanpa label. Pertama, saya membersihkan data, meng-encode kategori, membuang outlier, menyamakan skala, lalu memakai K-Means untuk membentuk 2 cluster (silhouette 0,57). Label cluster itu saya pakai untuk melatih Decision Tree dan Random Forest, yang mencapai akurasi 100%.*
>
> *Tapi saat saya selidiki, ternyata kedua cluster hanya membagi kota berdasarkan urutan abjad. Kolom `Location` di-LabelEncode menjadi 0–42 tanpa diskalakan, sehingga memegang 95,7% variansi dan mendominasi jarak K-Means. Itu juga yang membuat klasifikasinya sempurna — model cukup menghafal 21 kota. Kalau hanya memakai fitur perilaku, silhouette-nya turun ke 0,24, tapi cluster-nya jauh lebih bermakna: transaksi besar, nasabah senior bersaldo tinggi, dan nasabah muda bersaldo rendah.*
>
> *Pelajaran utamanya: encoding harus sesuai jenis data, semua fitur yang masuk ke algoritma berbasis jarak harus diskalakan, dan metrik yang bagus tidak menjamin hasil yang bermakna."*

---

## ✅ Daftar Periksa Akhir

Kamu sudah memahami proyek ini luar dalam jika bisa menjelaskan — tanpa melihat catatan:

- [ ] Tujuan proyek dan kenapa ada dua tahap (Bab 1)
- [ ] Beda supervised/unsupervised, fitur/label, nominal/ordinal (Bab 2)
- [ ] Pola `fit` / `transform` / `predict` dan kenapa versi library penting (Bab 3)
- [ ] Kondisi data mentah dan nasib setiap kolom (Bab 4)
- [ ] Setiap langkah notebook clustering beserta catatan kritisnya (Bab 5)
- [ ] Enam bukti bahwa cluster ditentukan oleh `Location` (Bab 6)
- [ ] Cara kerja Decision Tree, Random Forest, metrik, dan GridSearchCV (Bab 7)
- [ ] Kenapa akurasi 100% bukan prestasi dan bukan overfitting (Bab 7)
- [ ] Cara memperbaiki proyek dan kenapa silhouette tertinggi tidak selalu terbaik (Bab 8)

---

**← Sebelumnya:** [Bab 8 — Eksperimen dan Perbaikan](08-eksperimen-dan-perbaikan.md) · **Kembali ke:** [Daftar Isi](README.md)
