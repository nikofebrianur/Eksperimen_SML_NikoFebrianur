import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump
import os


def preprocess_data(data, target_column, save_path, output_csv_path):
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

    # Gabungkan ke ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Pisahkan fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Fit dan transform data
    X_processed = preprocessor.fit_transform(X)

    # Simpan pipeline
    dump(preprocessor, save_path)
    print(f"Pipeline berhasil disimpan ke: {save_path}")

    # Konversi hasil ke DataFrame (jika memungkinkan)
    try:
        X_processed_df = pd.DataFrame(X_processed.toarray())
    except:
        X_processed_df = pd.DataFrame(X_processed)

    X_processed_df[target_column] = y.values

    # Simpan hasil akhir ke CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    X_processed_df.to_csv(output_csv_path, index=False)
    print(f"Data siap latih berhasil disimpan ke: {output_csv_path}")


if __name__ == "__main__":
    df = pd.read_csv("dataset_raw/zara.csv", delimiter=";")
    preprocess_data(
        data=df,
        target_column="Sales Volume",
        save_path="preprocessing/preprocessor_pipeline.joblib",
        output_csv_path="preprocessing/dataset_preprocessing/zara_ready.csv"
    )
