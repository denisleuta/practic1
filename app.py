import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing import load_data
from som_clustering import train_som
from linear_model import train_linear_model
from visualizations import plot_happiness_histogram, plot_som_map, plot_map

# Загружаем данные
data = load_data()

# Обучаем модель и кластеризацию
features = ["Score", "Cost of Living Index", "AQI Value"]
som, clusters = train_som(data, features)
data["Cluster"] = clusters

# Запускаем Streamlit интерфейс
st.title("Анализ стран по счастью, стоимости жизни и загрязнению воздуха")

# Фильтры
min_score = st.sidebar.slider("Минимальный уровень счастья", min_value=0.0, max_value=10.0, value=5.0)
filtered_data = data[data["Score"] >= min_score]

# Гистограмма уровня счастья
st.subheader("Гистограмма уровня счастья")
plot_happiness_histogram(filtered_data)

# Карта Кохонена
st.subheader("Карта Кохонена: Кластеры стран")
plot_som_map(som, data, clusters)

# Генерация карты с использованием Plotly
st.subheader("Географическое распределение стран")
plot_map(data, filtered_data)

# Модель предсказания счастья
model, predicted_scores = train_linear_model(data)
st.write(f"Модель обучена, пример предсказания: {predicted_scores[0]:.2f}")