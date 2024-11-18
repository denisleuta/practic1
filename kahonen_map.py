import pandas as pd

# Загрузка данных из Excel
file_path = 'Online Retail.xlsx'
data = pd.read_excel(file_path)

# Просмотр первых нескольких строк данных
print(data.head())