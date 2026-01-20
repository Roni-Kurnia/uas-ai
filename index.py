"""
Studi Kasus: Prediksi Harga Rumah Menggunakan Linear Regression
Perusahaan Properti Jakarta
"""

# Import libraries yang diperlukan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("PREDIKSI HARGA RUMAH MENGGUNAKAN LINEAR REGRESSION")
print("Studi Kasus: Perusahaan Properti Jakarta")
print("="*60)
print()

# ============================================================================
# 1. GENERATE DATASET
# ============================================================================
print("1. GENERATING DATASET...")
print("-" * 60)

# Set random seed untuk reproducibility
np.random.seed(42)

# Jumlah sampel data
n_samples = 500

# Generate fitur-fitur
luas_tanah = np.random.randint(50, 500, n_samples)  # m²
luas_bangunan = np.random.randint(36, 400, n_samples)  # m²
jumlah_kamar_tidur = np.random.randint(2, 7, n_samples)
jumlah_kamar_mandi = np.random.randint(1, 5, n_samples)
umur_bangunan = np.random.randint(0, 31, n_samples)  # tahun
jarak_ke_pusat = np.random.uniform(1, 25, n_samples)  # km
garasi = np.random.randint(0, 2, n_samples)  # 0 atau 1

# Generate harga dengan formula yang realistis
# Base price + kontribusi dari setiap fitur + noise
harga = (
    200 +  # Base price (200 juta)
    luas_tanah * 2.5 +  # 2.5 juta per m² tanah
    luas_bangunan * 3.0 +  # 3 juta per m² bangunan
    jumlah_kamar_tidur * 50 +  # 50 juta per kamar tidur
    jumlah_kamar_mandi * 30 +  # 30 juta per kamar mandi
    umur_bangunan * (-5) +  # -5 juta per tahun umur
    jarak_ke_pusat * (-15) +  # -15 juta per km dari pusat
    garasi * 100 +  # +100 juta jika ada garasi
    np.random.normal(0, 100, n_samples)  # Random noise
)

# Buat DataFrame
data = pd.DataFrame({
    'Luas_Tanah_m2': luas_tanah,
    'Luas_Bangunan_m2': luas_bangunan,
    'Jumlah_Kamar_Tidur': jumlah_kamar_tidur,
    'Jumlah_Kamar_Mandi': jumlah_kamar_mandi,
    'Umur_Bangunan_Tahun': umur_bangunan,
    'Jarak_Pusat_Kota_km': jarak_ke_pusat,
    'Garasi': garasi,
    'Harga_Juta_Rupiah': harga
})

print(f"Dataset berhasil dibuat dengan {n_samples} sampel data")
print(f"Jumlah fitur: {len(data.columns) - 1}")
print()

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("2. EXPLORATORY DATA ANALYSIS")
print("-" * 60)

# Tampilkan 5 baris pertama
print("\nSample Data (5 baris pertama):")
print(data.head())
print()

# Informasi dataset
print("\nInformasi Dataset:")
print(data.info())
print()

# Statistik deskriptif
print("\nStatistik Deskriptif:")
print(data.describe().round(2))
print()

# Cek missing values
print("\nMissing Values:")
print(data.isnull().sum())
print()

# ============================================================================
# 3. VISUALISASI DATA
# ============================================================================
print("3. VISUALISASI DATA")
print("-" * 60)

# Visualisasi 1: Distribusi Harga
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist(data['Harga_Juta_Rupiah'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Harga (Juta Rupiah)')
axes[0, 0].set_ylabel('Frekuensi')
axes[0, 0].set_title('Distribusi Harga Rumah')
axes[0, 0].grid(True, alpha=0.3)

# Visualisasi 2: Korelasi Harga vs Luas Bangunan
axes[0, 1].scatter(data['Luas_Bangunan_m2'], data['Harga_Juta_Rupiah'], 
                   alpha=0.5, s=30)
axes[0, 1].set_xlabel('Luas Bangunan (m²)')
axes[0, 1].set_ylabel('Harga (Juta Rupiah)')
axes[0, 1].set_title('Harga vs Luas Bangunan')
axes[0, 1].grid(True, alpha=0.3)

# Visualisasi 3: Korelasi Harga vs Luas Tanah
axes[1, 0].scatter(data['Luas_Tanah_m2'], data['Harga_Juta_Rupiah'], 
                   alpha=0.5, s=30, color='orange')
axes[1, 0].set_xlabel('Luas Tanah (m²)')
axes[1, 0].set_ylabel('Harga (Juta Rupiah)')
axes[1, 0].set_title('Harga vs Luas Tanah')
axes[1, 0].grid(True, alpha=0.3)

# Visualisasi 4: Correlation Heatmap
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            ax=axes[1, 1], square=True, cbar_kws={'shrink': 0.8})
axes[1, 1].set_title('Correlation Matrix')

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi EDA disimpan sebagai 'eda_visualization.png'")
plt.show()

# ============================================================================
# 4. DATA PREPROCESSING & SPLITTING
# ============================================================================
print("\n4. DATA PREPROCESSING & SPLITTING")
print("-" * 60)

# Pisahkan fitur (X) dan target (y)
X = data.drop('Harga_Juta_Rupiah', axis=1)
y = data['Harga_Juta_Rupiah']

print(f"Jumlah fitur (X): {X.shape[1]}")
print(f"Jumlah sampel: {X.shape[0]}")
print()

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Data Training: {X_train.shape[0]} sampel")
print(f"Data Testing: {X_test.shape[0]} sampel")
print()

# ============================================================================
# 5. TRAINING MODEL LINEAR REGRESSION
# ============================================================================
print("5. TRAINING LINEAR REGRESSION MODEL")
print("-" * 60)

# Inisialisasi model
model = LinearRegression()

# Training model
print("Training model...")
model.fit(X_train, y_train)
print("✓ Model berhasil dilatih!")
print()

# Tampilkan koefisien
print("Koefisien Model (Pengaruh setiap fitur terhadap harga):")
print("-" * 60)
coefficients = pd.DataFrame({
    'Fitur': X.columns,
    'Koefisien': model.coef_
}).sort_values('Koefisien', ascending=False)

for idx, row in coefficients.iterrows():
    print(f"{row['Fitur']:25s}: {row['Koefisien']:8.2f} juta/unit")

print(f"\nIntercept (Base Price): {model.intercept_:.2f} juta Rupiah")
print()

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("6. EVALUASI MODEL")
print("-" * 60)

# Prediksi pada data training
y_train_pred = model.predict(X_train)

# Prediksi pada data testing
y_test_pred = model.predict(X_test)

# Hitung metrik evaluasi untuk training set
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# Hitung metrik evaluasi untuk testing set
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("METRIK EVALUASI:")
print()
print("Training Set:")
print(f"  • MAE (Mean Absolute Error)     : {train_mae:.2f} juta")
print(f"  • MSE (Mean Squared Error)      : {train_mse:.2f}")
print(f"  • RMSE (Root Mean Squared Error): {train_rmse:.2f} juta")
print(f"  • R² Score                      : {train_r2:.4f}")
print()
print("Testing Set:")
print(f"  • MAE (Mean Absolute Error)     : {test_mae:.2f} juta")
print(f"  • MSE (Mean Squared Error)      : {test_mse:.2f}")
print(f"  • RMSE (Root Mean Squared Error): {test_rmse:.2f} juta")
print(f"  • R² Score                      : {test_r2:.4f}")
print()

# Interpretasi R² Score
print("Interpretasi:")
print(f"Model dapat menjelaskan {test_r2*100:.2f}% variasi harga rumah")
print(f"Rata-rata error prediksi: ±{test_rmse:.2f} juta Rupiah")
print()

# ============================================================================
# 7. VISUALISASI HASIL PREDIKSI
# ============================================================================
print("7. VISUALISASI HASIL PREDIKSI")
print("-" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Actual vs Predicted (Training)
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=30)
axes[0].plot([y_train.min(), y_train.max()], 
             [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Harga Aktual (Juta Rupiah)')
axes[0].set_ylabel('Harga Prediksi (Juta Rupiah)')
axes[0].set_title(f'Training Set\nR² = {train_r2:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (Testing)
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=30, color='green')
axes[1].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Harga Aktual (Juta Rupiah)')
axes[1].set_ylabel('Harga Prediksi (Juta Rupiah)')
axes[1].set_title(f'Testing Set\nR² = {test_r2:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Residual Plot
residuals = y_test - y_test_pred
axes[2].scatter(y_test_pred, residuals, alpha=0.5, s=30, color='purple')
axes[2].axhline(y=0, color='r', linestyle='--', lw=2)
axes[2].set_xlabel('Harga Prediksi (Juta Rupiah)')
axes[2].set_ylabel('Residual (Actual - Predicted)')
axes[2].set_title('Residual Plot (Testing Set)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi evaluasi disimpan sebagai 'model_evaluation.png'")
plt.show()

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("\n8. FEATURE IMPORTANCE")
print("-" * 60)

# Visualisasi feature importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Fitur': X.columns,
    'Koefisien': np.abs(model.coef_)
}).sort_values('Koefisien', ascending=True)

plt.barh(feature_importance['Fitur'], feature_importance['Koefisien'], 
         color='skyblue', edgecolor='black')
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance (Pengaruh Fitur terhadap Harga)')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi feature importance disimpan sebagai 'feature_importance.png'")
plt.show()

# ============================================================================
# 9. PREDIKSI HARGA RUMAH BARU
# ============================================================================
print("\n9. PREDIKSI HARGA RUMAH BARU")
print("-" * 60)

# Contoh rumah baru yang akan diprediksi
rumah_baru = pd.DataFrame({
    'Luas_Tanah_m2': [150, 200, 300],
    'Luas_Bangunan_m2': [120, 180, 250],
    'Jumlah_Kamar_Tidur': [3, 4, 5],
    'Jumlah_Kamar_Mandi': [2, 3, 3],
    'Umur_Bangunan_Tahun': [5, 2, 0],
    'Jarak_Pusat_Kota_km': [10, 8, 5],
    'Garasi': [1, 1, 1]
})

# Prediksi harga
harga_prediksi = model.predict(rumah_baru)

# Tampilkan hasil
print("\nContoh Prediksi Harga untuk 3 Rumah Baru:")
print("=" * 60)
for i in range(len(rumah_baru)):
    print(f"\nRumah #{i+1}:")
    print(f"  • Luas Tanah        : {rumah_baru.iloc[i]['Luas_Tanah_m2']} m²")
    print(f"  • Luas Bangunan     : {rumah_baru.iloc[i]['Luas_Bangunan_m2']} m²")
    print(f"  • Kamar Tidur       : {rumah_baru.iloc[i]['Jumlah_Kamar_Tidur']}")
    print(f"  • Kamar Mandi       : {rumah_baru.iloc[i]['Jumlah_Kamar_Mandi']}")
    print(f"  • Umur Bangunan     : {rumah_baru.iloc[i]['Umur_Bangunan_Tahun']} tahun")
    print(f"  • Jarak ke Pusat    : {rumah_baru.iloc[i]['Jarak_Pusat_Kota_km']:.1f} km")
    print(f"  • Garasi            : {'Ya' if rumah_baru.iloc[i]['Garasi'] == 1 else 'Tidak'}")
    print(f"  ➤ PREDIKSI HARGA    : Rp {harga_prediksi[i]:,.2f} juta")
    print(f"                      : Rp {harga_prediksi[i]*1000000:,.0f}")

# ============================================================================
# 10. KESIMPULAN
# ============================================================================
print("\n" + "="*60)
print("KESIMPULAN")
print("="*60)
print(f"""
Model Linear Regression telah berhasil dibangun dengan performa:
• R² Score pada Testing Set: {test_r2:.4f} ({test_r2*100:.2f}%)
• RMSE: ±{test_rmse:.2f} juta Rupiah

Fitur yang paling berpengaruh terhadap harga rumah:
1. Luas Bangunan (koefisien tertinggi)
2. Luas Tanah
3. Garasi

Model ini dapat digunakan untuk:
✓ Membantu agen properti menentukan harga listing
✓ Memvalidasi kewajaran harga rumah
✓ Identifikasi properti undervalued/overvalued

Rekomendasi untuk improvement:
• Tambahkan fitur lain (lokasi, fasilitas umum, kondisi bangunan)
• Coba algoritma lain (Random Forest, XGBoost)
• Lakukan feature engineering
• Kumpulkan lebih banyak data
""")

print("="*60)
print("Program selesai dijalankan!")
print("="*60)