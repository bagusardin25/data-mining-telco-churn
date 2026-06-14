# Tugas Besar Penambangan Data — Prediksi & Segmentasi Customer Churn

Proyek ini merupakan Tugas Besar mata kuliah **BBK2LAB3 – Penambangan Data** yang menganalisis dataset **Telco Customer Churn** untuk memprediksi pelanggan yang berisiko churn dan memahami segmen pelanggan melalui teknik data mining.

## Deskripsi Proyek

Industri telekomunikasi menghadapi tingkat *customer churn* (pelanggan berhenti berlangganan) yang tinggi. Proyek ini menerapkan prinsip-prinsip Penambangan Data menggunakan metodologi **CRISP-DM** untuk:

1. **Segmentasi pelanggan** menggunakan **K-Means Clustering** (unsupervised learning) — mengidentifikasi 3 segmen pelanggan dengan karakteristik dan tingkat churn yang berbeda.
2. **Prediksi churn** menggunakan **Logistic Regression** dan **Naïve Bayes** (supervised learning) — memprediksi pelanggan yang berisiko churn dengan ROC AUC ~0.84.
3. **Dashboard interaktif** menggunakan **Streamlit** — menyajikan insight, visualisasi, dan fitur prediksi individual.

## Dataset

- **Sumber**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7.043 pelanggan, 21 atribut
- **Target**: `Churn` (Yes/No) — proporsi churn ~26.5%

## Struktur Proyek

```
data-mining-telco-churn/
├── data/
│   ├── raw/                          # Dataset asli
│   └── processed/                    # Data setelah preprocessing
├── src/
│   ├── 01_eda.py                     # Eksplorasi data & visualisasi
│   ├── 02_preprocessing.py           # Cleaning, encoding, scaling, split
│   ├── 03_clustering_kmeans.py       # K-Means clustering
│   ├── 04_classification.py          # Logistic Regression & Naïve Bayes
│   └── 05_evaluation.py             # Evaluasi & perbandingan model
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── models/                           # Model tersimpan (.pkl)
├── reports/
│   └── figures/                      # Visualisasi hasil analisis
├── requirements.txt
└── README.md
```

## Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Pipeline Analisis
```bash
python src/01_eda.py              # Eksplorasi data
python src/02_preprocessing.py    # Preprocessing
python src/03_clustering_kmeans.py # K-Means clustering
python src/04_classification.py   # Klasifikasi
python src/05_evaluation.py       # Evaluasi
```

### 3. Jalankan Dashboard
```bash
streamlit run dashboard/app.py
```

## Hasil Utama

### Segmentasi (K-Means, k=3)
| Cluster | Jumlah | Tenure Avg | Monthly Charges Avg | Churn Rate |
|---------|--------|------------|---------------------|------------|
| 0       | 2.382  | 26.5 bln   | $28.89              | 14.4%      |
| 1       | 2.178  | 58.8 bln   | $89.33              | 14.6%      |
| 2       | 2.483  | 14.8 bln   | $77.62              | **48.7%**  |

### Klasifikasi (Model Terbaik: LogReg SMOTE)
| Metrik    | Nilai  |
|-----------|--------|
| Accuracy  | 0.758  |
| Precision | 0.531  |
| Recall    | 0.746  |
| F1-Score  | 0.621  |
| ROC AUC   | 0.835  |

## Tech Stack

- **Python 3.11** — pandas, numpy, scikit-learn, matplotlib, seaborn, plotly
- **imbalanced-learn** — SMOTE untuk penanganan data tidak seimbang
- **Streamlit** — Dashboard interaktif
