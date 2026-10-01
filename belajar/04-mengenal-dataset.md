# Bab 4 — Mengenal Dataset

> **Inti bab ini:** sebelum mengolah data, kenali dulu isinya. Bab ini membahas ke-16 kolom, kondisi data mentahnya (data hilang, duplikat, sebaran), dan "nasib" setiap kolom di proyek ini.

---

## 4.1 Asal Data

Dataset ini adalah versi modifikasi dari **"Bank Transaction Dataset for Fraud Detection"** (Kaggle). Dicoding menyediakan versinya sendiri lewat Google Sheets, dan notebook memuatnya langsung dari URL:

```python
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?output=csv'
df = pd.read_csv(url)
```

### Kenapa wajib versi Google Sheets, bukan Kaggle?

| | Versi Kaggle | Versi Google Sheets (dipakai) |
|---|---:|---:|
| Jumlah baris | 2.512 | **2.537** |
| Data hilang (missing value) | 0 | **ada di setiap kolom (18–30 per kolom)** |
| Baris duplikat | 0 | **21** |

Dicoding **sengaja menyisipkan** data hilang dan duplikat. Tujuannya supaya langkah `dropna()` dan `drop_duplicates()` benar-benar melakukan sesuatu, dan supaya output-mu bisa dicocokkan dengan "output yang diharapkan" di template (misalnya `duplicated().sum()` harus menghasilkan **21**). Kalau memakai versi Kaggle, angkanya jadi 0 dan tidak cocok.

---

## 4.2 Ke-16 Kolom dan Nasibnya

| # | Kolom | Jenis | Penjelasan | Nasib di proyek ini |
|---|---|---|---|---|
| 1 | `TransactionID` | ID | Kode unik transaksi | 🗑️ Dibuang |
| 2 | `AccountID` | ID | Kode unik akun | 🗑️ Dibuang |
| 3 | `TransactionAmount` | Numerik | Nominal transaksi | ✂️ Outlier dibuang, 📏 di-*scale* |
| 4 | `TransactionDate` | Tanggal | Waktu transaksi | 🗑️ Dibuang |
| 5 | `TransactionType` | Kategorikal (2) | `Credit` / `Debit` | 🔢 LabelEncoder |
| 6 | `Location` | Kategorikal (43) | Kota di Amerika Serikat | 🔢 LabelEncoder ⚠️ *(biang masalah, lihat Bab 6)* |
| 7 | `DeviceID` | ID | Kode perangkat | 🗑️ Dibuang |
| 8 | `IP Address` | ID | Alamat IP | 🗑️ Dibuang |
| 9 | `MerchantID` | ID | Kode merchant | 🗑️ Dibuang |
| 10 | `Channel` | Kategorikal (3) | `ATM` / `Branch` / `Online` | 🔢 LabelEncoder |
| 11 | `CustomerAge` | Numerik | Usia nasabah | 📏 di-*scale*, lalu 🪣 dikelompokkan jadi `CustomerAge_Group` |
| 12 | `CustomerOccupation` | Kategorikal (4) | `Doctor` / `Engineer` / `Retired` / `Student` | 🔢 LabelEncoder |
| 13 | `TransactionDuration` | Numerik | Lama transaksi (detik) | 📏 di-*scale* |
| 14 | `LoginAttempts` | Numerik (diskrit) | Jumlah percobaan login | ✂️ Outlier dibuang → jadi konstan |
| 15 | `AccountBalance` | Numerik | Saldo setelah transaksi | 📏 di-*scale* |
| 16 | `PreviousTransactionDate` | Tanggal | Tanggal transaksi sebelumnya | 🗑️ Dibuang |

Setelah semua langkah, tersisa **9 kolom asli + 1 kolom baru** (`CustomerAge_Group`) = 10 fitur untuk K-Means.

---

## 4.3 Kondisi Data Mentah

### Data hilang per kolom

| Kolom | Jumlah kosong | | Kolom | Jumlah kosong |
|---|---:|---|---|---:|
| TransactionID | 29 | | Channel | 27 |
| AccountID | 21 | | CustomerAge | 18 |
| TransactionAmount | 26 | | CustomerOccupation | 23 |
| PreviousTransactionDate | 28 | | TransactionDuration | 26 |
| TransactionType | 30 | | LoginAttempts | 21 |
| Location | 30 | | AccountBalance | 27 |
| DeviceID | 30 | | TransactionDate | 24 |
| IP Address | 20 | | MerchantID | 23 |

Kalau dijumlahkan per kolom totalnya 403 sel kosong, tapi yang penting adalah **berapa baris yang punya minimal satu sel kosong: 381 baris**. Itulah jumlah baris yang hilang ketika kita menjalankan `dropna()` (sekitar **15%** data).

> 💡 Kenapa `CustomerAge` di data ini bertipe `float64` (angka desimal) padahal umur itu bilangan bulat? Karena di pandas, sel kosong (`NaN`) adalah nilai desimal. Satu saja `NaN` di sebuah kolom membuat seluruh kolom ikut menjadi `float64`. Di versi Kaggle (tanpa data hilang), kolom ini bertipe `int64`.

### Statistik kolom numerik (data mentah)

| Kolom | Rata-rata | Std | Min | Median | Maks |
|---|---:|---:|---:|---:|---:|
| TransactionAmount | 297,66 | 292,23 | 0,26 | 211,36 | 1.919,11 |
| CustomerAge | 44,68 | 17,84 | 18 | 45 | 80 |
| TransactionDuration | 119,42 | 70,08 | 10 | 112 | 300 |
| LoginAttempts | 1,12 | 0,59 | 1 | 1 | 5 |
| AccountBalance | 5.113,44 | 3.897,98 | 101,25 | 4.734,11 | 14.977,99 |

Hal-hal menarik dari tabel ini:

- **`TransactionAmount` miring ke kanan** (*right-skewed*): rata-rata (297,66) jauh di atas median (211,36). Artinya sebagian besar transaksi kecil, dan ada segelintir transaksi sangat besar (sampai 1.919,11) yang menarik rata-rata ke atas.
- **Rentang angka antar kolom sangat berbeda.** Saldo bisa sampai belasan ribu, umur maksimal 80. Inilah kenapa penskalaan diperlukan (ingat contoh jarak di Bab 2).

### `LoginAttempts` hampir selalu 1

| Nilai | Jumlah baris |
|---|---:|
| 1 | 2.396 |
| 2 | 27 |
| 3 | 31 |
| 4 | 31 |
| 5 | 31 |
| kosong | 21 |

Sekitar 95% transaksi hanya butuh 1 kali login. Transaksi dengan 2–5 kali percobaan login itu **jarang** — dan justru karena jarang, dalam konteks keamanan bank, itulah sinyal yang **paling menarik** (bisa jadi percobaan pembobolan). Ingat ini saat membaca tahap pembuangan outlier di Bab 5.

### Sebaran kolom kategorikal (data mentah, tanpa yang kosong)

| Kolom | Sebaran |
|---|---|
| TransactionType | Debit 1.942 · Credit 565 |
| Channel | Branch 868 · ATM 836 · Online 806 |
| CustomerOccupation | Student 657 · Doctor 633 · Engineer 627 · Retired 597 |
| Location | 43 kota berbeda |

Sebaran `Channel` dan `CustomerOccupation` sangat **merata**. Artinya, tidak ada satu jenis nasabah yang dominan — petunjuk awal bahwa mencari "persona" yang tegas dari kolom-kolom ini akan sulit.

### Korelasi antar kolom numerik

| | Amount | Age | Duration | Login | Balance |
|---|---:|---:|---:|---:|---:|
| **TransactionAmount** | 1,00 | -0,02 | 0,00 | -0,01 | -0,03 |
| **CustomerAge** | -0,02 | 1,00 | -0,01 | 0,01 | **0,32** |
| **TransactionDuration** | 0,00 | -0,01 | 1,00 | 0,03 | 0,01 |
| **LoginAttempts** | -0,01 | 0,01 | 0,03 | 1,00 | 0,01 |
| **AccountBalance** | -0,03 | **0,32** | 0,01 | 0,01 | 1,00 |

Hanya ada **satu hubungan yang lumayan**: semakin tua nasabah, cenderung semakin besar saldonya (korelasi 0,32). Kolom lain hampir tidak berhubungan satu sama lain (korelasi mendekati 0).

> 💡 **Cara membaca korelasi:** nilainya antara -1 dan 1. Mendekati **1** = naik bersama; mendekati **-1** = yang satu naik, yang lain turun; mendekati **0** = tidak ada hubungan *linear*. Korelasi 0,32 tergolong lemah-sedang.

---

## 4.4 Ringkasan: Apa yang Kita Ketahui Sebelum Memodelkan

1. Ada **data hilang** di setiap kolom dan **21 duplikat** → perlu dibersihkan.
2. Ada **7 kolom ID/tanggal** yang tidak berguna untuk mencari pola umum → perlu dibuang.
3. Ada **4 kolom kategorikal** (salah satunya `Location` dengan 43 nilai) → perlu diubah jadi angka.
4. **Skala kolom numerik sangat berbeda** → perlu disamakan.
5. **Sebaran kategori sangat merata** dan **korelasi antar kolom sangat lemah** → data ini mungkin tidak punya kelompok alami yang tegas.

Poin ke-5 adalah petunjuk penting yang baru benar-benar terbukti di Bab 6 dan Bab 8.

---

**← Sebelumnya:** [Bab 3 — Alat dan Lingkungan](03-alat-dan-lingkungan.md) · **Selanjutnya:** [Bab 5 — Bedah Notebook Clustering →](05-notebook-clustering.md)
