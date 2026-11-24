import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Установка случайного зерна для воспроизводимости
np.random.seed(42)

# Генерация данных
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Hannah', 'Ian', 'Jack'],
    'Age': [25, 30, np.nan, 22, 28, np.nan, 35, 40, 29, 31],
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York', np.nan, 'Chicago', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles'],
    'Score': [np.nan, 88, 92, 85, 90, 95, np.nan, 80, 75, 85]
}

# Создание DataFrame
df = pd.DataFrame(data)

# Вывод исходного DataFrame
print("Исходный DataFrame:")
print(df)

# 1. Удаление строк с пропусками
df_dropped = df.dropna()
print("\nDataFrame после удаления строк с пропусками:")
print(df_dropped)

# 2. Заполнение пропусков
# Заполним пропуски в 'Age' средним значением, в 'City' - наиболее частым значением, в 'Score' - медианой
df_filled = df.copy()
df_filled['Age'].fillna(df_filled['Age'].mean(), inplace=True)
df_filled['City'].fillna(df_filled['City'].mode()[0], inplace=True)
df_filled['Score'].fillna(df_filled['Score'].median(), inplace=True)

print("\nDataFrame после заполнения пропусков:")
print(df_filled)

# 3. Кодирование строковых значений с помощью LabelEncoder
label_encoder = LabelEncoder()
df_filled['City'] = label_encoder.fit_transform(df_filled['City'])

print("\nDataFrame после кодирования строковых значений:")
print(df_filled)
