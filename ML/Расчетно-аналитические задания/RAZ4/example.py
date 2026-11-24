import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Создание набора данных
data = {
    'Цвет': ['Красный', 'Красный', 'Желтый', 'Желтый', 'Красный', 'Желтый'],
    'Размер': ['Маленький', 'Большой', 'Средний', 'Маленький', 'Средний', 'Большой'],
    'Класс': ['Яблоко', 'Яблоко', 'Банан', 'Банан', 'Яблоко', 'Банан']
}

# Преобразование в DataFrame
df = pd.DataFrame(data)

# Преобразование категориальных переменных в числовые
df['Цвет'] = df['Цвет'].astype('category').cat.codes
df['Размер'] = df['Размер'].astype('category').cat.codes
df['Класс'] = df['Класс'].astype('category').cat.codes

# Разделение данных на признаки и целевую переменную
X = df[['Цвет', 'Размер']]
y = df['Класс']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели дерева решений
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

# Вывод дерева решений
tree_rules = export_text(model, feature_names=['Цвет', 'Размер'])
print(tree_rules)

# Визуализация дерева решений
plt.figure(figsize=(10, 6))
tree.plot_tree(model, feature_names=['Цвет', 'Размер'], class_names=['Яблоко', 'Банан'], filled=True)
plt.show()

# Оценка модели
accuracy = model.score(X_test, y_test)
print(f'Точность модели: {accuracy:.2f}')
