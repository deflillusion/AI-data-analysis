import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Анализ транзакций", layout="wide")
st.title("Интерактивный анализ транзакций")

uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Универсальное приведение типов для совместимости с Streamlit (Arrow)
    df = df.convert_dtypes()

    for col in df.columns:
        try:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype("float64")
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype("bool")
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype("string")
        except Exception:
            df[col] = df[col].astype("string")

    st.write("📊 Размерность данных:", df.shape)

    st.subheader("📌 Пример данных")
    st.dataframe(df.head())

    st.subheader("🔍 Типы данных")
    st.write(df.dtypes)

    st.subheader("🚫 Пропущенные значения")
    st.write(df.isnull().sum())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.subheader("📈 Распределение числовых признаков")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Распределение: {col}")
            st.pyplot(fig)

    if len(numeric_cols) > 1:
        st.subheader("🧠 Кластеризация клиентов")
        num_data = df[numeric_cols].dropna()
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(num_data)
        df['Cluster'] = -1
        df.loc[num_data.index, 'Cluster'] = clusters

        st.write("🎯 Результаты кластеризации")
        st.dataframe(df[['Cluster'] + numeric_cols].head())

        st.subheader("🗺️ Визуализация кластеров")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=df[numeric_cols[0]],
            y=df[numeric_cols[1]],
            hue=df['Cluster'],
            palette="viridis",
            ax=ax
        )
        ax.set_title("Кластеры по двум признакам")
        st.pyplot(fig)

else:
    st.info("⬆️ Загрузите CSV-файл для начала анализа.")
