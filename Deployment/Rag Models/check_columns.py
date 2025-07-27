import pandas as pd

df = pd.read_parquet("app/passages.parquet")
print("📋 Available Columns:")
print(df.columns)
print("🔍 First Few Rows:")
print(df.head())