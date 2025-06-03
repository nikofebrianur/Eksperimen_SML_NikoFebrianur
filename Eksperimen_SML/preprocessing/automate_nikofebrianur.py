import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

def preprocess_data(data, target_column, save_path_pipeline, output_dir):
    # Menentukan fitur numerik dan kategorikal
    numeric_features = ['price', 'Sales Volume']
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    # Pastikan kolom target tidak termasuk
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Pipeline untuk numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk kategorikal
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabungkan semua pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Pisahkan fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Fit dan transform fitur
    X_processed = preprocessor.fit_transform(X)

    # Simpan pipeline
    os.makedirs(os.path.dirname(save_path_pipeline), exist_ok=True)
    dump(preprocessor, save_path_pipeline)
    print(f"Pipeline berhasil disimpan ke: {save_path_pipeline}")

    # Konversi hasil ke DataFrame (jika OneHotEncoder menghasilkan sparse matrix)
    try:
        X_processed_df = pd.DataFrame(X_processed.toarray())
    except:
        X_processed_df = pd.DataFrame(X_processed)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed_df, y, test_size=0.2, random_state=42
    )

    # Simpan hasil split
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"Data training dan testing berhasil disimpan ke folder: {output_dir}")

# Eksekusi utama
if __name__ == "__main__":
    df = pd.read_csv("Eksperimen_SML/dataset_raw/zara.csv", delimiter=";")
    preprocess_data(
        data=df,
        target_column="Sales Volume",
        save_path_pipeline="Eksperimen_SML/preprocessing/preprocessor_pipeline.joblib",
        output_dir="Eksperimen_SML/preprocessing/dataset_split"
    )
