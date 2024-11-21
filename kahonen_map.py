import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---- Загрузка данных ----
@st.cache_data
def load_data():
    # Основные датасеты
    happiness = pd.read_csv('2019.csv')
    cost_of_living = pd.read_csv('Cost_of_Living_Index_by_Country_2024.csv')
    pollution = pd.read_csv('global_air_pollution_dataset.csv')
    coordinates = pd.read_csv('World Cities Nearest Stations.csv')

    # Предобработка данных
    happiness.rename(columns={'Country or region': 'Country'}, inplace=True)
    coordinates.rename(columns={'country': 'Country', 'lat': 'latitude', 'lng': 'longitude'}, inplace=True)

    # Удаление дубликатов
    happiness = happiness.drop_duplicates(subset='Country')
    cost_of_living = cost_of_living.drop_duplicates(subset='Country')
    pollution = pollution.drop_duplicates(subset='Country')
    coordinates = coordinates.drop_duplicates(subset='Country')

    # Объединение данных
    data = (
        happiness
        .merge(cost_of_living, on='Country', how='inner')
        .merge(pollution, on='Country', how='inner')
        .merge(coordinates, on='Country', how='left')  # Добавляем координаты
    )

    # Удаление строк с пропущенными значениями
    data.dropna(inplace=True)
    return data

data = load_data()

# ---- Нормализация данных ----
@st.cache_data
def normalize_data(data, features):
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
    return normalized

# ---- Кластеризация (Карта Кохонена) ----
@st.cache_resource
def train_som(data, features, som_size=(10, 10)):
    normalized_data = normalize_data(data, features).values

    # Инициализация карты
    som = MiniSom(x=som_size[0], y=som_size[1], input_len=len(features), sigma=1.0, learning_rate=0.5)
    som.random_weights_init(normalized_data)

    # Обучение
    som.train_random(data=normalized_data, num_iteration=1000)

    # Присвоение кластеров
    clusters = [som.winner(row) for row in normalized_data]
    return som, clusters

features = ['Score', 'Cost of Living Index', 'AQI Value']
som, clusters = train_som(data, features)
data['Cluster'] = clusters

# ---- Интерфейс Streamlit ----
st.title("Анализ данных о счастье, стоимости жизни и загрязнении")

# ---- Фильтры ----
st.sidebar.header("Фильтры")
min_score = st.sidebar.slider(
    "Минимальный уровень счастья",
    min_value=float(data['Score'].min()),
    max_value=float(data['Score'].max()),
    value=float(data['Score'].min())
)
filtered_data = data[data['Score'] >= min_score]

# ---- Карта ----
st.subheader("Карта стран участвующих в сравнении по координатам")
if 'latitude' in data.columns and 'longitude' in data.columns:
    st.map(filtered_data[['latitude', 'longitude']])
else:
    st.warning("Координаты отсутствуют в данных!")

# ---- Таблица данных ----
st.subheader("Фильтрованные данные")
st.dataframe(filtered_data)

# ---- Гистограмма уровня счастья ----
st.subheader("Гистограмма уровня счастья")
st.bar_chart(filtered_data[['Country', 'Score']].set_index('Country'))

# ---- Кластеризация (визуализация карты Кохонена) ----
st.subheader("Карта Кохонена: Кластеры стран")
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Карта расстояний между нейронами")
ax.imshow(som.distance_map().T, cmap='coolwarm', interpolation='nearest')
for i, (x, y) in enumerate(data['Cluster']):
    ax.text(x, y, data['Country'].iloc[i], fontsize=8, ha='center', va='center')
st.pyplot(fig)

# ---- Тепловая карта корреляций ----
st.subheader("Тепловая карта корреляций")
correlation_features = ['Score', 'Cost of Living Index', 'AQI Value', 'GDP per capita', 'Social support', 'Healthy life expectancy']
corr_matrix = data[correlation_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Корреляции между показателями")
st.pyplot(fig)

# ---- Диаграмма зависимости счастья от стоимости жизни ----
st.subheader("Зависимость уровня счастья от стоимости жизни")
fig, ax = plt.subplots()
scatter = ax.scatter(
    data['Cost of Living Index'], data['Score'],
    c=data['AQI Value'], cmap='coolwarm', edgecolor='k', alpha=0.7
)
fig.colorbar(scatter, ax=ax, label='AQI Value')
ax.set_xlabel("Cost of Living Index")
ax.set_ylabel("Happiness Score")
ax.set_title("Happiness vs Cost of Living")
st.pyplot(fig)

# ---- Топ-10 стран ----
st.subheader("Топ-10 стран для переезда")
data['Total Score'] = normalize_data(data, features).sum(axis=1)
top_countries = data[['Country', 'Total Score']].sort_values(by='Total Score', ascending=False).head(10)

fig, ax = plt.subplots()
top_countries.plot(kind='bar', x='Country', y='Total Score', ax=ax, legend=False, color='skyblue')
ax.set_ylabel("Total Score")
ax.set_title("Топ-10 стран для переезда")
st.pyplot(fig)
