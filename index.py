"""
PREDIKSI HARGA RUMAH - LINEAR REGRESSION 
Studi Kasus: Perusahaan Properti Jakarta

Kelompok UAS Artificial Intelligence:
Nama: Roni Kurnia
NIM: 3420230002

Nama : Ahmad Fauzan
NIM : 3420230018
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("="*50)
print("PREDIKSI HARGA RUMAH")
print("="*50)

# ========================================
# 1. BUAT DATA
# ========================================
print("\n1. Membuat Dataset...")

np.random.seed(42)
n = 100  # 100 data rumah

# Buat data fitur
data = pd.DataFrame({
    'Luas_Tanah': np.random.randint(50, 300, n),
    'Luas_Bangunan': np.random.randint(36, 200, n),
    'Jumlah_Kamar': np.random.randint(2, 6, n),
    'Umur_Bangunan': np.random.randint(0, 20, n),
})

# Hitung harga (formula sederhana)
data['Harga'] = (
    data['Luas_Tanah'] * 2 + 
    data['Luas_Bangunan'] * 3 + 
    data['Jumlah_Kamar'] * 50 - 
    data['Umur_Bangunan'] * 5 +
    np.random.randint(-50, 50, n)
)

print(f"✓ Data dibuat: {n} rumah")
print("\nContoh 5 data pertama:")
print(data.head())

# ========================================
# 2. PISAHKAN DATA
# ========================================
print("\n2. Memisahkan Data Training dan Testing...")

X = data[['Luas_Tanah', 'Luas_Bangunan', 'Jumlah_Kamar', 'Umur_Bangunan']]
y = data['Harga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✓ Data Training: {len(X_train)} rumah")
print(f"✓ Data Testing: {len(X_test)} rumah")

# ========================================
# 3. LATIH MODEL
# ========================================
print("\n3. Melatih Model Linear Regression...")

model = LinearRegression()
model.fit(X_train, y_train)

print("✓ Model berhasil dilatih!")
print("\nKoefisien (pengaruh tiap fitur):")
for i, col in enumerate(X.columns):
    print(f"  {col:20s}: {model.coef_[i]:6.2f}")

# ========================================
# 4. EVALUASI MODEL
# ========================================
print("\n4. Evaluasi Model...")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✓ MAE (error rata-rata): {mae:.2f} juta")
print(f"✓ R² Score: {r2:.4f} ({r2*100:.1f}%)")

# ========================================
# 5. VISUALISASI
# ========================================
print("\n5. Membuat Visualisasi...")

plt.figure(figsize=(10, 4))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Harga Aktual')
plt.ylabel('Harga Prediksi')
plt.title('Actual vs Predicted')
plt.grid(True, alpha=0.3)

# Plot 2: Feature Importance
plt.subplot(1, 2, 2)
features = X.columns
importance = np.abs(model.coef_)
plt.barh(features, importance)
plt.xlabel('Koefisien')
plt.title('Pengaruh Fitur')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hasil_prediksi.png', dpi=150)
print("✓ Grafik disimpan: hasil_prediksi.png")
plt.show()

# ========================================
# 6. PREDIKSI RUMAH BARU
# ========================================
print("\n6. Prediksi Harga Rumah Baru...")
print("-"*50)

# Contoh 2 rumah baru
rumah_baru = pd.DataFrame({
    'Luas_Tanah': [150, 250],
    'Luas_Bangunan': [100, 180],
    'Jumlah_Kamar': [3, 4],
    'Umur_Bangunan': [5, 2]
})

harga_prediksi = model.predict(rumah_baru)

for i in range(len(rumah_baru)):
    print(f"\nRumah #{i+1}:")
    print(f"  Luas Tanah    : {rumah_baru.iloc[i]['Luas_Tanah']} m²")
    print(f"  Luas Bangunan : {rumah_baru.iloc[i]['Luas_Bangunan']} m²")
    print(f"  Jumlah Kamar  : {rumah_baru.iloc[i]['Jumlah_Kamar']}")
    print(f"  Umur Bangunan : {rumah_baru.iloc[i]['Umur_Bangunan']} tahun")
    print(f"  → PREDIKSI    : Rp {harga_prediksi[i]:.2f} juta")

# ========================================
# KESIMPULAN
# ========================================
print("\n" + "="*50)
print("KESIMPULAN")
print("="*50)
print(f"""
Model dapat memprediksi harga dengan akurasi {r2*100:.1f}%
Error rata-rata: ±{mae:.2f} juta

Fitur paling berpengaruh:
→ Luas Bangunan (koefisien: {model.coef_[1]:.2f})
→ Luas Tanah (koefisien: {model.coef_[0]:.2f})
""")
print("="*50)
print("Program selesai!")
print("="*50)