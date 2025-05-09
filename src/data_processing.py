import pandas as pd
import streamlit as st
from openai import OpenAI


def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".xml"):
            df = pd.read_xml(uploaded_file)
        else:
            st.error("Неподдерживаемый формат файла.")
            return None

        # Проверка на дублирующиеся колонки
        if df.columns.duplicated().any():
            st.error(
                f"Обнаружены дублирующиеся колонки в датасете: {df.columns[df.columns.duplicated()].tolist()}")
            df = df.loc[:, ~df.columns.duplicated()]
            st.warning("Дублирующиеся колонки удалены.")

        return df
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        return None


def classify_columns(df, client):
    column_types = {}
    sample_data = df.head(10).to_dict(orient='records')

    for col in df.columns:
        prompt = f"""
        Ты — аналитик данных. Я даю тебе название колонки и примеры её значений из банковского датасета.
        Колонка: "{col}"
        Примеры значений: {sample_data[:5]}
        Классифицируй колонку как один из типов: 
        - идентификатор (например, user_id, client_id — идентификатор клиента)
        - дата (например, transaction_date)
        - сумма (например, amount, value)
        - категория (например, category, merchant_type)
        - текст (например, description, merchant_name)
        - другое (например, transaction_id — ID транзакции)
        Ответь только тип колонки.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            column_type = response.choices[0].message.content.strip()
            column_types[col] = column_type if column_type in [
                'идентификатор', 'дата', 'сумма', 'категория', 'текст', 'другое'] else 'другое'
            st.write(f"Колонка '{col}' классифицирована как: {column_type}")
        except Exception as e:
            st.warning(f"Ошибка классификации колонки {col}: {e}")
            column_types[col] = 'другое'

    return column_types
