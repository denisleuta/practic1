import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
import streamlit as st

# Загрузка данных
happiness = pd.read_csv('2019.csv')
cost_of_living = pd.read_csv('Cost_of_Living_Index_by_Country_2024.csv')  # Данные стоимости жизни по странам
pollution = pd.read_csv('global_air_pollution_dataset.csv')  # Загрязнение воздуха по странам

# Переименование столбца в датасете Happiness
happiness.rename(columns={'Country or region': 'Country'}, inplace=True)

# Удаление дубликатов по странам
happiness = happiness.drop_duplicates(subset='Country')
cost_of_living = cost_of_living.drop_duplicates(subset='Country')
pollution = pollution.drop_duplicates(subset='Country')

# Объединение по странам
merged_data = happiness.merge(cost_of_living, on='Country').merge(pollution, on='Country')

# Удаление строк с пропусками
merged_data.dropna(inplace=True)

# Сохранение объединённых данных
merged_data.to_csv('merged_country_data.csv', index=False)
print("Данные успешно объединены!")

# Выбор интересующих признаков
features = ['Score', 'Cost of Living Index', 'AQI Value']

# Нормализация
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(
    scaler.fit_transform(merged_data[features]),
    columns=features
)

# Сохранение нормализованных данных
normalized_data.to_csv('normalized_country_data.csv', index=False)
print("Данные нормализованы!")

# Параметры карты
som_size = (10, 10)  # Размер карты
som = MiniSom(x=som_size[0], y=som_size[1], input_len=len(features), sigma=1.0, learning_rate=0.5)

# Инициализация весов
som.random_weights_init(normalized_data.values)

# Обучение
som.train_random(data=normalized_data.values, num_iteration=1000)
print("Карта Кахонена обучена!")

# Присваивание кластеров странам
clusters = [som.winner(row) for row in normalized_data.values]
merged_data['Cluster'] = clusters
merged_data.to_csv('clustered_country_data.csv', index=False)

# Карта расстояний
plt.figure(figsize=(10, 10))
plt.title("Карта Кахонена: Расстояния между нейронами")
plt.imshow(som.distance_map().T, cmap='coolwarm')  # Карта расстояний
plt.colorbar(label='Distance')
plt.show()

plt.figure(figsize=(10, 10))
for i, (x, y) in enumerate(merged_data['Cluster']):
    plt.text(x, y, merged_data['Country'].iloc[i], fontsize=8)
plt.imshow(som.distance_map().T, cmap='coolwarm')
plt.colorbar(label='Distance')
plt.title("Страны по кластерам")
plt.show()

# Печать первых нескольких строк для проверки
st.write("Данные о счастье:")
st.write(happiness.head())
st.write("Данные о стоимости жизни:")
st.write(cost_of_living.head())
st.write("Данные о загрязнении:")
st.write(pollution.head())

coordinates_data = pd.read_csv('World Cities Nearest Stations.csv')
coordinates_data.rename(columns={'country': 'Country'}, inplace=True)

# Слияние данных по ключу "Country"
happiness = happiness.merge(coordinates_data, on='Country', how='left')

# Слияние остальных данных
merged_data = (
    happiness
    .merge(cost_of_living, left_on='Country', right_on='Country', how='inner')
    .merge(pollution, left_on='Country', right_on='Country', how='inner')
)

# Переименование столбцов для карты
merged_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'}, inplace=True)

# Заголовок приложения
st.title("Анализ данных счастья, стоимости жизни и загрязнения")

# Фильтрация данных
st.sidebar.header("Фильтры")
min_score = st.sidebar.slider("Минимальный уровень счастья", 
                               min_value=float(merged_data['Score'].min()), 
                               max_value=float(merged_data['Score'].max()), 
                               value=float(merged_data['Score'].min()))
filtered_data = merged_data[merged_data['Score'] >= min_score]

filtered_data = filtered_data.drop_duplicates(subset='Country')

# Отображение таблицы
st.subheader("Фильтрованные данные")
st.dataframe(filtered_data)

# Отображение карты
st.subheader("Карта данных")
st.map(filtered_data[['latitude', 'longitude']])

# Визуализация статистики
st.subheader("Гистограмма уровня счастья")
st.bar_chart(filtered_data[['Country', 'Score']].set_index('Country'))