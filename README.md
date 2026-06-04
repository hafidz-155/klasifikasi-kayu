# Klasifikasi Jenis Kayu Menggunakan GLCM dan KNN

## Deskripsi
Aplikasi ini digunakan untuk mengidentifikasi jenis kayu berdasarkan tekstur citra menggunakan metode Gray Level Co-occurrence Matrix (GLCM) dan algoritma K-Nearest Neighbor (KNN).

NAMA : HAFIDZ HASAN SUNANDAR
NRP : 152024155

## Fitur
- Upload gambar kayu
- Preprocessing citra
- Ekstraksi fitur GLCM
- Klasifikasi menggunakan KNN
- Menampilkan hasil prediksi jenis kayu

## Teknologi
- Python
- OpenCV
- NumPy
- Scikit-Learn
- Streamlit (jika menggunakan Streamlit)

## Cara Menjalankan

1. Clone repository

```bash
git clone https://github.com/hafidz-155/klasifikasi-kayu.git
```

2. Install dependency

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi

```bash
streamlit run app.py
```

## Metode

### Preprocessing
- Resize gambar
- Konversi grayscale

### Ekstraksi Fitur
Menggunakan fitur GLCM:
- Contrast
- Correlation
- Energy
- Homogeneity

### Klasifikasi
Menggunakan algoritma K-Nearest Neighbor (KNN).

## Hasil
Sistem dapat mengklasifikasikan beberapa jenis kayu berdasarkan tekstur citra yang diinputkan pengguna.

## Repository
https://github.com/hafidz-155/klasifikasi-kayu