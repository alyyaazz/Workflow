import pandas as pd
import numpy as np
import argparse
import os
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(data_path):
    """Fungsi untuk memuat dan membersihkan data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Preprocessing: Pilih hanya kolom numerik (misal: age, bmi, children, charges)
    # Karena KMeans memerlukan input numerik
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Handling missing values jika ada
    df_numeric = df_numeric.dropna()
    
    # Scaling data agar fitur memiliki rentang yang sama
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    return df, scaled_data

def train_model(data_path, n_clusters):
    """Fungsi utama untuk training dan logging ke MLflow."""
    
    # Memulai session MLflow
    with mlflow.start_run():
        # 1. Load data
        original_df, processed_data = load_and_preprocess(data_path)
        
        # 2. Inisialisasi Model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # 3. Training
        kmeans.fit(processed_data)
        
        # 4. Logging Parameter ke MLflow
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("input_file", data_path)
        
        # 5. Logging Metric (Inertia/WCSS untuk mengevaluasi cluster)
        mlflow.log_metric("inertia", kmeans.inertia_)
        
        # 6. Simpan Model sebagai Artefak
        mlflow.sklearn.log_model(kmeans, "cluster_model")
        
        # 7. Simpan Hasil Prediksi ke CSV dan log sebagai artefak tambahan
        original_df['cluster_label'] = kmeans.labels_
        output_name = "insurance_clustered_result.csv"
        original_df.to_csv(output_name, index=False)
        mlflow.log_artifact(output_name)
        
        print("-" * 30)
        print(f"Berhasil melatih model dengan {n_clusters} cluster.")
        print(f"Inertia Score: {kmeans.inertia_}")
        print(f"Hasil disimpan sebagai artefak: {output_name}")
        print("-" * 30)

if __name__ == "__main__":
    # Setup Argument Parser agar bisa dipanggil oleh file MLProject
    parser = argparse.ArgumentParser(description="Insurance Clustering Model")
    parser.add_argument("--data_path", type=str, default="insurance.csv", help="Path file dataset")
    parser.add_argument("--n_clusters", type=int, default=3, help="Jumlah cluster")
    
    args = parser.parse_args()
    
    # Jalankan proses training
    try:
        train_model(args.data_path, args.n_clusters)
    except Exception as e:
        print(f"Terjadi Kesalahan: {e}")