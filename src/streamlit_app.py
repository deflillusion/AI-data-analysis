import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import os
from dotenv import load_dotenv

# Настройка страницы
st.set_page_config(page_title="Анализ данных транзакций", layout="wide")
st.title("📊 Универсальный анализ транзакционных данных")

# Загрузка API-ключа
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API-ключ OpenAI не найден. Убедитесь, что он указан в файле .env.")
    st.stop()
client = OpenAI(api_key=api_key)

# Кэширование данных


@st.cache_data
def load_and_process_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".xml"):
            return pd.read_xml(uploaded_file)
        else:
            st.error("Неподдерживаемый формат файла.")
            return None
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        return None

# Функция анализа GPT


@st.cache_data
def generate_gpt_analysis(df, sample_size=100):
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    sample_csv = sample_df.to_csv(index=False)
    prompt = f"""
    Ты — аналитик данных. Перед тобой таблица с транзакциями пользователей.
    Проанализируй, какие поведенческие или финансовые особенности видны.
    Предложи 3–5 идей для графиков, которые можно построить.
    Вот первые строки данных в CSV-формате:
    {sample_csv}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Ты — эксперт по анализу данных."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Ошибка при обращении к GPT: {e}")
        return None


# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл (.csv, .xlsx, .xml)", type=[
                                 "csv", "xlsx", "xls", "xml"])
if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
    if df is not None:
        st.success("Файл успешно загружен и прочитан.")
        st.subheader("📄 Предпросмотр данных")
        st.dataframe(df.head())

        # Информация о колонках
        st.subheader("📌 Информация о типах данных")
        dtypes_df = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "num_unique": [df[col].nunique() for col in df.columns],
            "num_missing": [df[col].isna().sum() for col in df.columns],
        })
        st.dataframe(dtypes_df)

        # Пользовательский график
        st.subheader("📊 Построить пользовательский график")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        x_col = st.selectbox("Выберите колонку для X", cat_cols)
        y_col = st.selectbox("Выберите колонку для Y", numeric_cols)
        if st.button("Построить график"):
            fig, ax = plt.subplots()
            df.groupby(x_col)[y_col].sum().plot(kind='bar', ax=ax)
            ax.set_title(f"Сумма {y_col} по {x_col}")
            st.pyplot(fig)

        # GPT-анализ
        if st.button("🔍 Получить идеи от GPT"):
            with st.spinner("Анализируем с помощью GPT..."):
                gpt_result = generate_gpt_analysis(df)
                if gpt_result:
                    st.subheader("💡 Идеи и анализ от GPT:")
                    st.markdown(gpt_result)
else:
    st.info("Пожалуйста, загрузите файл для анализа.")
