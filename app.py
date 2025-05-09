import streamlit as st
import pandas as pd
from src.data_processing import load_data, classify_columns
from src.feature_engineering import create_features
from src.modeling import train_kmeans
from src.utils import export_to_csv, export_to_excel
from openai import OpenAI
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы
st.set_page_config(page_title="Гибкий анализ транзакций", layout="wide")
st.title("📊 Гибкий анализ транзакций и сегментация клиентов")

# Инициализация session_state для хранения графика
if 'plot_data' not in st.session_state:
    st.session_state['plot_data'] = None
if 'cluster_summary' not in st.session_state:
    st.session_state['cluster_summary'] = None
if 'features_df' not in st.session_state:
    st.session_state['features_df'] = None

# Загрузка API-ключа
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API-ключ OpenAI не найден. Укажите его в файле .env.")
    st.stop()
client = OpenAI(api_key=api_key)

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл (.csv, .xlsx, .xml)", type=[
                                 "csv", "xlsx", "xls", "xml"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.success("Файл успешно загружен!")
        st.subheader("📄 Предпросмотр данных")
        st.dataframe(df.head())

        # Вывод уникальных значений для диагностики
        st.subheader("🔎 Информация о данных")
        for col in df.columns:
            st.write(
                f"Колонка '{col}': {df[col].nunique()} уникальных значений")

        # Классификация колонок
        st.subheader("🔍 Классификация колонок")
        if st.button("Классифицировать колонки с помощью GPT"):
            with st.spinner("Анализируем колонки..."):
                column_types = classify_columns(df, client)
                st.session_state['column_types'] = column_types
                st.write("Предложенная классификация колонок:")
                st.json(column_types)

        # Ручная корректировка классификации
        if 'column_types' in st.session_state:
            st.subheader("✏️ Корректировка классификации")
            column_types = st.session_state['column_types']
            for col in df.columns:
                column_types[col] = st.selectbox(
                    f"Тип колонки '{col}'",
                    options=['идентификатор', 'дата', 'сумма',
                             'категория', 'текст', 'другое'],
                    index=['идентификатор', 'дата', 'сумма', 'категория',
                           'текст', 'другое'].index(column_types[col])
                )
            st.session_state['column_types'] = column_types

            # Кластеризация
            st.subheader("🤖 Поведенческая сегментация")
            features_df = create_features(df, column_types)
            if features_df is None or len(features_df) == 0:
                st.error(
                    "Не удалось создать признаки. Проверьте данные и классификацию колонок.")
            else:
                n_samples = len(features_df)
                max_clusters = min(n_samples, 10)
                if n_samples < 2:
                    st.error(
                        "Недостаточно данных для кластеризации. Нужно хотя бы 2 уникальных клиента.")
                else:
                    n_clusters = st.slider(
                        "Выберите количество кластеров",
                        2,
                        max_clusters,
                        min(4, max_clusters)
                    )
                    if st.button("Запустить кластеризацию"):
                        with st.spinner("Выполняется кластеризация..."):
                            try:
                                kmeans, scaler, X_scaled = train_kmeans(
                                    features_df, n_clusters)
                                features_df['cluster'] = kmeans.predict(
                                    X_scaled)
                                st.session_state['features_df'] = features_df

                                # Визуализация
                                st.subheader("📊 Визуализация кластеров")
                                fig, ax = plt.subplots()
                                sns.scatterplot(
                                    data=features_df, x='pca_1', y='pca_2', hue='cluster', palette='viridis', ax=ax)
                                ax.set_title("Сегментация клиентов (PCA)")
                                st.pyplot(fig)
                                # Сохраняем график
                                st.session_state['plot_data'] = fig

                                # Характеристики кластеров
                                st.subheader("📈 Характеристики кластеров")
                                cluster_summary = features_df.groupby(
                                    'cluster').mean()
                                st.dataframe(cluster_summary)
                                st.session_state['cluster_summary'] = cluster_summary

                            except ValueError as e:
                                st.error(
                                    f"Ошибка кластеризации: {e}. Проверьте данные и классификацию колонок.")

                    # Интерпретация от GPT
                    if st.session_state['cluster_summary'] is not None:
                        st.subheader("🤖 Интерпретация кластеров")
                        # Повторно отображаем график, если он был создан
                        if st.session_state['plot_data'] is not None:
                            st.subheader("📊 Визуализация кластеров (повтор)")
                            st.pyplot(st.session_state['plot_data'])

                        if st.button("Получить интерпретацию от GPT"):
                            with st.spinner("Запрашиваем интерпретацию..."):
                                try:
                                    st.write("Отправляем запрос к GPT...")
                                    prompt = f"""
                                    Ты — аналитик данных. Вот средние значения признаков для кластеров клиентов:
                                    {st.session_state['cluster_summary'].to_csv()}
                                    Опиши, какие типы клиентов соответствуют каждому кластеру. Дай краткое название сегменту.
                                    """
                                    response = client.chat.completions.create(
                                        model="gpt-3.5-turbo",  # Заменили на более доступную модель
                                        messages=[
                                            {"role": "user", "content": prompt}],
                                        temperature=0.7,
                                        max_tokens=500
                                    )
                                    st.write("Ответ от GPT получен.")
                                    st.markdown(
                                        "### 🤖 Интерпретация кластеров")
                                    st.markdown(
                                        response.choices[0].message.content)
                                except Exception as e:
                                    st.error(
                                        f"Ошибка при запросе к GPT: {e}. Проверьте API-ключ или подключение.")

                        # Экспорт
                        if st.session_state['features_df'] is not None:
                            st.subheader("💾 Экспорт результатов")
                            csv_data = export_to_csv(
                                st.session_state['features_df'])
                            st.download_button("Скачать результаты (CSV)",
                                               csv_data,
                                               "segmentation_results.csv",
                                               "text/csv")

                            excel_data = export_to_excel(
                                st.session_state['features_df'])
                            st.download_button("Скачать результаты (Excel)",
                                               excel_data,
                                               "segmentation_results.xlsx",
                                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
