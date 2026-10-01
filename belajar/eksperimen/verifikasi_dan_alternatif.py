# -*- coding: utf-8 -*-
"""
Skrip pendamping Bab 6 dan Bab 8 panduan belajar.

Bagian A  — membuktikan bahwa cluster di notebook ditentukan oleh kolom Location.
Bagian B  — mencoba cara clustering alternatif yang lebih bermakna.

Cara menjalankan (dari folder utama repo, dengan environment bmlp_env aktif):
    python belajar/eksperimen/verifikasi_dan_alternatif.py

Skrip ini HANYA membaca file data/model. Tidak ada file submission yang diubah.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data = pd.read_csv(os.path.join(REPO, "data_clustering.csv"))           # versi ter-encode & ter-scale
inv = pd.read_csv(os.path.join(REPO, "data_clustering_inverse.csv"))    # versi nilai asli
X = data.drop(columns="Target")


def judul(teks):
    print("\n" + "=" * 70 + f"\n{teks}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# BAGIAN A — Apa yang sebenarnya memisahkan kedua cluster?
# ---------------------------------------------------------------------------
judul("A1. Standar deviasi & porsi variansi setiap fitur (input K-Means)")
var = X.var()
tabel = pd.DataFrame({"std": X.std().round(2), "porsi_variansi_%": (var / var.sum() * 100).round(1)})
print(tabel.sort_values("porsi_variansi_%", ascending=False))

judul("A2. Pusat cluster (centroid) dari model_clustering.h5")
kmeans = joblib.load(os.path.join(REPO, "model_clustering.h5"))
print(pd.DataFrame(kmeans.cluster_centers_, columns=X.columns, index=["Cluster 0", "Cluster 1"]).T.round(2))
print("\nkmeans.predict(X) sama dengan kolom Target:", (kmeans.predict(X) == data["Target"]).mean() == 1.0)

judul("A3. Rentang kode Location di setiap cluster")
print(data.groupby("Target")["Location"].agg(["min", "max"]))
kota = sorted(inv["Location"].unique())
print("Cluster 0:", ", ".join(kota[:22]))
print("Cluster 1:", ", ".join(kota[22:]))
silang = pd.crosstab(inv["Location"], inv["Target"])
print("Jumlah kota yang muncul di kedua cluster:", int(((silang[0] > 0) & (silang[1] > 0)).sum()))

judul("A4. PCA: seberapa besar komponen pertama dan apa isinya")
pca = PCA(n_components=2).fit(X)
print("Variansi yang dijelaskan:", np.round(pca.explained_variance_ratio_, 4))
print("Bobot PCA1 terbesar:")
print(pd.Series(pca.components_[0], index=X.columns).abs().sort_values(ascending=False).head(3).round(3))

judul("A5. Uji klasifikasi: dengan vs tanpa Location")
y = inv["Target"]


def akurasi(fitur):
    Xf = pd.get_dummies(fitur, drop_first=True)
    Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, random_state=42, stratify=y)
    model = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
    return accuracy_score(yte, model.predict(Xte))


print(f"Semua fitur           : {akurasi(inv.drop(columns='Target')):.4f}")
print(f"Hanya Location        : {akurasi(inv[['Location']]):.4f}")
print(f"Semua kecuali Location: {akurasi(inv.drop(columns=['Target', 'Location'])):.4f}")

judul("A6. Proporsi profesi & kelompok umur per cluster (%)")
for kolom in ["CustomerOccupation", "CustomerAge_Group"]:
    print(f"\n{kolom}")
    print((pd.crosstab(inv[kolom], inv["Target"], normalize="columns") * 100).round(1))

# ---------------------------------------------------------------------------
# BAGIAN B — Alternatif yang lebih bermakna
# ---------------------------------------------------------------------------
numerik = ["TransactionAmount", "CustomerAge", "TransactionDuration", "AccountBalance"]

judul("B1. Hanya fitur perilaku numerik (diskalakan), tanpa Location")
Xa = StandardScaler().fit_transform(inv[numerik])
for k in range(2, 7):
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xa)
    print(f"  k={k}  silhouette={silhouette_score(Xa, lab):.4f}")

lab3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Xa)
profil = inv[numerik].assign(cluster=lab3).groupby("cluster").mean().round(1)
profil["jumlah"] = np.bincount(lab3)
print("\nProfil k=3 (rata-rata dalam satuan asli):")
print(profil)

judul("B2. Fitur numerik diskalakan + kategorikal one-hot")
kategori = ["TransactionType", "Location", "Channel", "CustomerOccupation"]
Xb = pd.concat(
    [pd.DataFrame(StandardScaler().fit_transform(inv[numerik]), columns=numerik),
     pd.get_dummies(inv[kategori]).astype(float)],
    axis=1,
)
print("Jumlah fitur:", Xb.shape[1])
for k in range(2, 7):
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xb)
    print(f"  k={k}  silhouette={silhouette_score(Xb, lab):.4f}")

print("\nSelesai.")
