import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st


def create_features(df, column_types):
    # Очистка данных
    df = df.copy()

    # Определяем ключевые колонки
    id_col = next((col for col, ctype in column_types.items()
                  if ctype == 'идентификатор'), None)
    date_col = next((col for col, ctype in column_types.items()
                    if ctype == 'дата'), None)
    amount_col = next(
        (col for col, ctype in column_types.items() if ctype == 'сумма'), None)
    category_cols = [col for col, ctype in column_types.items()
                     if ctype == 'категория']

    if not id_col:
        st.error(
            "Не найден идентификатор клиента (например, client_id). Кластеризация невозможна.")
        return None

    # Проверка количества уникальных клиентов
    n_unique_ids = df[id_col].nunique()
    if n_unique_ids < 2:
        st.error(
            f"Найден только {n_unique_ids} уникальный клиент. Для кластеризации нужно хотя бы 2 клиента.")
        return None

    # Отладка: выводим идентификатор
    st.write(
        f"Используется колонка идентификатора: {id_col} ({n_unique_ids} уникальных значений)")

    # Преобразование даты
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

    # Инициализация features
    features = pd.DataFrame({id_col: df[id_col].unique()})

    # Числовые агрегации
    if amount_col:
        agg_funcs = {
            amount_col: ['mean', 'median', 'sum', 'min', 'max', 'count']
        }
        user_agg = df.groupby(id_col).agg(agg_funcs).reset_index()
        user_agg.columns = [f"{col[0]}_{col[1]}" if col[1]
                            else col[0] for col in user_agg.columns]
        features = features.merge(user_agg, on=id_col, how='left')

        # Частота транзакций
        if date_col:
            date_agg = df.groupby(id_col)[date_col].agg(
                ['min', 'max']).reset_index()
            date_agg['days_active'] = (
                date_agg['max'] - date_agg['min']).dt.days + 1
            date_agg['tx_per_day'] = features[f"{amount_col}_count"] / \
                date_agg['days_active'].replace(0, 1)
            features = features.merge(
                date_agg[[id_col, 'days_active', 'tx_per_day']], on=id_col, how='left')

    # Категориальные признаки
    for cat_col in category_cols:
        category_pivot = pd.pivot_table(df,
                                        index=id_col,
                                        columns=cat_col,
                                        values=amount_col if amount_col else id_col,
                                        aggfunc='sum' if amount_col else 'count',
                                        fill_value=0)
        category_pivot = category_pivot.div(category_pivot.sum(axis=1), axis=0)
        category_pivot.columns = [
            f"{cat_col}_share_{col}" for col in category_pivot.columns]
        category_pivot = category_pivot.reset_index()
        features = features.merge(category_pivot, on=id_col, how='left')

    # Проверка на дублирующиеся колонки
    if features.columns.duplicated().any():
        st.error(
            f"Обнаружены дублирующиеся колонки в features: {features.columns[features.columns.duplicated()].tolist()}")
        features = features.loc[:, ~features.columns.duplicated()]

    # Отладка: выводим созданные колонки
    st.write(f"Созданы признаки: {features.columns.tolist()}")

    # PCA для визуализации
    numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(features[numeric_cols].fillna(0))
        features['pca_1'] = pca_features[:, 0]
        features['pca_2'] = pca_features[:, 1]

    return features
