# universal_analyzer.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_file(filepath):
    df = pd.read_csv(filepath)

    print(f"Строк: {len(df)}, Колонок: {len(df.columns)}")
    print("\nТипы данных:")
    print(df.dtypes)

    print("\nПропуски по колонкам:")
    print(df.isnull().sum())

    print("\nПример данных:")
    print(df.head())

    # Попробуем найти похожие на важные поля:
    candidates = {
        "amount": [c for c in df.columns if 'amount' in c.lower() or 'sum' in c.lower()],
        "date": [c for c in df.columns if 'date' in c.lower()],
        "category": [c for c in df.columns if 'cat' in c.lower()],
        "client_id": [c for c in df.columns if 'client' in c.lower() or 'user' in c.lower()],
    }
    print("\nВозможные ключевые поля:")
    for key, cols in candidates.items():
        print(f"  {key}: {cols}")

    # График распределения суммы (если найдена)
    if candidates["amount"]:
        col = candidates["amount"][0]
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f'Распределение: {col}')
        plt.show()

    # Группировка по категории
    if candidates["category"]:
        cat_col = candidates["category"][0]
        print("\nТоп 10 категорий:")
        print(df[cat_col].value_counts().head(10))
