from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
import streamlit as st


def train_kmeans(features_df, n_clusters):
    # Проверка на дублирующиеся колонки
    if features_df.columns.duplicated().any():
        st.error(
            f"Обнаружены дублирующиеся колонки перед кластеризацией: {features_df.columns[features_df.columns.duplicated()].tolist()}")
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]

    # Подготовка данных
    numeric_cols = features_df.select_dtypes(
        include=['float64', 'int64']).columns
    X = features_df[numeric_cols].fillna(0)

    # Проверка количества образцов
    if len(X) < n_clusters:
        raise ValueError(
            f"Количество клиентов ({len(X)}) меньше числа кластеров ({n_clusters}). Уменьшите количество кластеров.")

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Обучение K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Создание папки models/, если она не существует
    os.makedirs('models', exist_ok=True)

    # Сохранение модели и скейлера
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    return kmeans, scaler, X_scaled
