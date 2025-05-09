import pandas as pd
import io


def export_to_csv(df):
    return df.to_csv(index=False)


def export_to_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()
