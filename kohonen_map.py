import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClusterMixin
from minisom import MiniSom
from sklearn_som.som import SOM
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from PIL import Image

# ---- Заголовок и визуальное оформление ----
st.set_page_config(page_title="Анализ стран по счастью и качеству жизни", layout="wide")
st.title("Анализ стран по счастью, стоимости жизни и загрязнению воздуха")

# Загружаем изображение фона
image = Image.open("background.jpg")
st.image(image, use_container_width=True)

# ---- Введение ----
st.markdown(
    """
    Этот анализ позволяет исследовать связи между уровнями счастья, стоимостью жизни и загрязнением воздуха в различных странах.
    Мы используем несколько показателей, чтобы оценить общие условия жизни и создать рекомендации по переезду в зависимости от выбранных критериев.
    С помощью фильтров можно сузить выборку стран по ключевым показателям и увидеть визуальные результаты анализа.
"""
)

# ---- Загрузка данных ----
@st.cache_data
def load_data():
    # Основные датасеты
    happiness = pd.read_csv("data/happiness_2019.csv")
    cost_of_living = pd.read_csv("data/Cost_of_Living_Index_by_Country_2024.csv")
    pollution = pd.read_csv("data/global_air_pollution_dataset.csv")
    coordinates = pd.read_csv("data/World Cities Nearest Stations.csv")

    additional_data = pd.read_csv("data/merged_country_data.csv")

    # Предобработка данных
    happiness.rename(columns={"Country or region": "Country"}, inplace=True)
    coordinates.rename(
        columns={"country": "Country", "lat": "latitude", "lng": "longitude"},
        inplace=True,
    )

    additional_data.rename(columns={"country_name": "Country"}, inplace=True)

    # Удаление дубликатов
    coordinates = coordinates.drop_duplicates(subset="Country")
    additional_data = additional_data.drop_duplicates(subset="Country")

    # Объединение данных
    data1 = (
        happiness.merge(cost_of_living, on="Country", how="inner")
        .merge(pollution, on="Country", how="inner")
        .merge(coordinates, on="Country", how="left")
    )

    data = additional_data.merge(coordinates, on="Country")

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


# ---- Веса для показателей ----
WEIGHTS = {
    "Score": 0.25,  # Уровень счастья
    "Cost of Living Index": 0.15,  # Стоимость жизни
    "GDP per capita": 0.2,  # ВВП населения
    "Healthy life expectancy": 0.2,  # Здоровье
    "Social support": 0.1,  # Социальная поддержка
    "PM2.5 AQI Value": 0.1,  # Качество воздуха
}

# ---- Кластеризация (Карта Кохонена) ----
@st.cache_resource
def train_som(data, features, som_size=(20, 20)):
    normalized_data = normalize_data(data, features).values

    # Инициализация карты
    som = MiniSom(
        x=som_size[0],
        y=som_size[1],
        input_len=len(features),
        sigma=1.0,
        learning_rate=0.5,
    )
    som.random_weights_init(normalized_data)

    # Обучение
    som.train_random(data=normalized_data, num_iteration=2000)

    # Присвоение кластеров
    clusters = [som.winner(row) for row in normalized_data]
    return som, clusters


features = ["Score", "Cost of Living Index", "PM2.5 AQI Value"]
som, clusters = train_som(data, features)
data["Cluster"] = clusters

# ---- Нормализация всех показателей ----
selected_features = list(WEIGHTS.keys())
normalized_data = normalize_data(data, selected_features)

# Класс для использования SOM с GridSearchCV
class SOMSklearn(BaseEstimator, ClusterMixin):
    def __init__(self, m=10, n=10, dim=3, epochs=100, shuffle=True):
        self.m = m
        self.n = n
        self.dim = dim
        self.epochs = epochs
        self.shuffle = shuffle

    def fit(self, X, y=None):
        self.som = SOM(m=self.m, n=self.n, dim=self.dim)
        self.som.fit(X, epochs=self.epochs, shuffle=self.shuffle)
        return self

    def predict(self, X):
        return self.som.predict(X)


# Функция для проведения гиперпараметрической оптимизации
def train_som_sklearn_with_optimization(data, features, param_grid, cv=5):
    normalized_data = normalize_data(data, features).values
    som_sklearn = SOMSklearn()

    # Настройка GridSearchCV для гиперпараметрической оптимизации
    grid_search = GridSearchCV(
        som_sklearn, param_grid, cv=cv, scoring="neg_mean_squared_error"
    )

    grid_search.fit(normalized_data)

    # Получаем лучшие параметры и модель
    best_params = grid_search.best_params_
    best_som = grid_search.best_estimator_

    clusters = best_som.predict(normalized_data)

    return best_som, clusters, best_params


param_grid = {
    "m": [10, 20],
    "n": [10, 20],
    "epochs": [100, 200],
    "shuffle": [True, False],
}

# Запуск с оптимизацией гиперпараметров
best_som, clusters_sklearn, best_params = train_som_sklearn_with_optimization(
    data, features, param_grid
)

data["Cluster_SKLEARN_OPTIMIZED"] = clusters_sklearn

# Применяем веса
for feature, weight in WEIGHTS.items():
    normalized_data[feature] *= weight

# Считаем итоговый рейтинг
data["Total Score"] = normalized_data.sum(axis=1)


# ---- Фильтры ----
st.sidebar.header("Фильтры")
st.sidebar.markdown(
    """
    Используйте фильтры для того, чтобы сузить выборку стран. Например, можно ограничить выбор стран по минимальному уровню счастья, стоимости жизни и качеству воздуха.
"""
)

# Фильтр по минимальному уровню счастья
min_score = st.sidebar.slider(
    "Минимальный уровень счастья",
    min_value=float(data["Score"].min()),
    max_value=float(data["Score"].max()),
    value=float(data["Score"].min()),
    help="Отфильтровать страны, где уровень счастья выше выбранного значения.",
)

# Фильтр по стоимости жизни (Cost of Living Index)
max_cost_of_living = st.sidebar.slider(
    "Максимальный индекс стоимости жизни",
    min_value=float(data["Cost of Living Index"].min()),
    max_value=float(data["Cost of Living Index"].max()),
    value=float(data["Cost of Living Index"].max()),
    help="Отфильтровать страны, где стоимость жизни ниже выбранного значения.",
)

# Фильтр по уровню здоровья (Healthy life expectancy)
min_health = st.sidebar.slider(
    "Минимальный индекс продолжительности здоровой жизни",
    min_value=float(data["Healthy life expectancy"].min()),
    max_value=float(data["Healthy life expectancy"].max()),
    value=float(data["Healthy life expectancy"].min()),
    help="Отфильтровать страны, где продолжительность здоровой жизни выше выбранного значения.",
)

# Фильтр по уровню загрязнения (PM2.5 AQI Value)
max_aqi = st.sidebar.slider(
    "Максимальный индекс загрязнения воздуха (AQI)",
    min_value=float(data["PM2.5 AQI Value"].min()),
    max_value=float(data["PM2.5 AQI Value"].max()),
    value=float(data["PM2.5 AQI Value"].max()),
    help="Отфильтровать страны с уровнем загрязнения ниже выбранного значения.",
)

# Применяем фильтры
filtered_data = data[
    (data["Score"] >= min_score)
    & (data["Cost of Living Index"] <= max_cost_of_living)
    & (data["Healthy life expectancy"] >= min_health)
    & (data["PM2.5 AQI Value"] <= max_aqi)
]

# ---- Метрики ----
st.sidebar.subheader("Основные метрики")
st.sidebar.metric(
    label="Средний индекс счастья", value=f"{filtered_data['Score'].mean():.2f}"
)

st.sidebar.metric(
    label="Средний индекс стоимости жизни",
    value=f"{filtered_data['Cost of Living Index'].mean():.2f}",
)

st.sidebar.metric(
    label="Средний индекс загрязнение воздуха",
    value=f"{filtered_data['AQI Value'].mean():.2f}",
)

# ---- Таблица данных ----
st.subheader("Фильтрованные данные")
st.markdown(
    """
    В таблице ниже отображаются все страны, которые удовлетворяют выбранным фильтрам.
    Вы можете увидеть их уровень счастья, стоимость жизни, продолжительность жизни и другие важные показатели.
"""
)
st.dataframe(filtered_data)

# ---- Гистограмма уровня счастья ----
st.subheader("Гистограмма уровня счастья")
st.markdown(
    """
    Гистограмма отображает распределение уровня счастья среди стран. Вы можете использовать фильтры, чтобы выбрать интересующие вас данные.
"""
)
st.bar_chart(filtered_data[["Country", "Score"]].set_index("Country"))

# ---- Кластеризация (визуализация карты Кохонена) ----
st.subheader("Карта Кохонена: Кластеры стран")
st.markdown(
    """
    Эта карта отображает кластеры стран на основе выбранных критериев. Страны с похожими показателями сгруппированы вместе.
"""
)
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Карта расстояний между нейронами")
ax.imshow(som.distance_map().T, cmap="coolwarm", interpolation="nearest")
for i, (x, y) in enumerate(data["Cluster"]):
    ax.text(x, y, data["Country"].iloc[i], fontsize=8, ha="center", va="center")
st.pyplot(fig)

# Визуализация кластеров из Sklearn-SOM с подписями стран
st.markdown("### Кластеры, полученные с помощью Sklearn-SOM")
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(
    data["latitude"],
    data["longitude"],
    c=data["Cluster_SKLEARN_OPTIMIZED"],
    cmap="viridis",
    s=100,
    alpha=0.7,
    edgecolors="k",
)

for i, row in data.iterrows():
    ax.text(
        row["latitude"],
        row["longitude"],
        row["Country"],
        fontsize=8,
        ha="center",
        va="center",
        color="black",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
    )

ax.set_title(
    "Sklearn-SOM: Географическое распределение кластеров",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Широта", fontsize=12)
ax.set_ylabel("Долгота", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Кластеры", fontsize=12)

st.pyplot(fig)

# Отображаем лучшие параметры
st.write(f"Лучшие параметры SOM: {best_params}")

st.markdown(
    """
### Как работать с картой:
1. **Точки** на карте представляют объекты стран.
2. **Цвет точки** соответствует определённому кластеру, присвоенному картой Кохонена.
3. **Координаты долготы и широты** соответсвует их географическому расположению.

### Сравнение карт:
- **MiniSom**: Быстрая реализация и подходит для небольших данных.
- **Sklearn-SOM**: Обеспечивает более гибкое использование благодаря интеграции с библиотекой scikit-learn.

"""
)

# ---- Генерация карты с использованием Plotly ----
fig_map = px.scatter_mapbox(
    filtered_data,
    lat="latitude",
    lon="longitude",
    color="Cluster",
    hover_name="Country",
    color_continuous_scale=px.colors.qualitative.Set1,
    zoom=2,
    height=600,
)

fig_map.update_layout(
    mapbox_style="carto-positron",
    title="Географическое распределение стран по кластерам",
)
st.plotly_chart(fig_map)

# ---- Тепловая карта корреляций ----
st.subheader("Тепловая карта корреляций")
st.markdown(
    """
    Тепловая карта отображает корреляцию между различными показателями (счастье, стоимость жизни, загрязнение воздуха и т.д.).
    Более темные цвета показывают более сильную корреляцию.
"""
)
corr_matrix = filtered_data[
    [
        "Score",
        "Cost of Living Index",
        "PM2.5 AQI Value",
        "GDP per capita",
        "Social support",
        "Healthy life expectancy",
    ]
].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Корреляции между показателями")
st.pyplot(fig)

# ---- Интерактивные графики с Plotly ----
st.subheader("Интерактивная диаграмма: Счастье и Стоимость жизни")
st.markdown(
    """
    На этой диаграмме показана зависимость уровня счастья от стоимости жизни. Цвет точек показывает уровень загрязнения воздуха.
"""
)
fig = px.scatter(
    filtered_data,
    x="Cost of Living Index",
    y="Score",
    color="PM2.5 AQI Value",
    hover_data=["Country"],
    title="Зависимость счастья от стоимости жизни",
)
st.plotly_chart(fig)

# ---- Топ-10 стран ----
st.subheader("Лидеры по качеству жизни: Топ-10 стран")
st.markdown(
    """
    На этом графике представлены 10 лучших стран по качеству жизни на основе общего рейтинга, который включает счастье, стоимость жизни и экологию.
"""
)
top_countries = (
    filtered_data[["Country", "Total Score"]]
    .sort_values(by="Total Score", ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    top_countries["Country"],
    top_countries["Total Score"],
    color=plt.cm.coolwarm(
        top_countries["Total Score"] / top_countries["Total Score"].max()
    ),
)

sm = plt.cm.ScalarMappable(
    cmap="coolwarm",
    norm=plt.Normalize(
        vmin=top_countries["Total Score"].min(), vmax=top_countries["Total Score"].max()
    ),
)
sm.set_array([])
fig.colorbar(
    sm, ax=ax, orientation="horizontal", fraction=0.02, pad=0.1, label="Total Score"
)

ax.set_xlabel("Total Score")
ax.set_title("Лидеры по качеству жизни: Топ-10 стран", fontsize=16, fontweight="bold")
ax.set_ylabel("Страна")

# Добавляем значения на барах
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}",
        ha="center",
        va="center",
        fontsize=12,
        color="black",
    )

plt.tight_layout()

st.pyplot(fig)

# ---- Модель предсказывающая уровень счастья населения ----

# Подготовка данных
X = data[
    [
        "Cost of Living Index",
        "PM2.5 AQI Value",
        "GDP per capita",
        "Social support",
        "Healthy life expectancy",
    ]
]
y = data["Score"]

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание
predicted_scores = model.predict(X_test)

# Добавление возможности предсказания через Streamlit
st.subheader("Предсказание уровня счастья")
st.markdown(
    """
    Введите значения для различных факторов, чтобы предсказать уровень счастья.
"""
)
cost_of_living = st.number_input(
    "Индекс стоимости жизни", min_value=0.0, max_value=100.0, value=50.0
)
aqi_value = st.number_input(
    "Индекс качества воздуха (PM2.5 AQI Value)",
    min_value=0.0,
    max_value=500.0,
    value=50.0,
)
gdp_per_capita = st.number_input("ВВП населения", min_value=0.0, value=1.3)
social_support = st.number_input(
    "Социальная поддержка", min_value=0.0, max_value=3.0, value=1.4
)
health_expectancy = st.number_input(
    "Ожидаемая продолжительность здоровой жизни",
    min_value=0.0,
    max_value=2.0,
    value=0.8,
)

predicted_score = model.predict(
    [[cost_of_living, aqi_value, gdp_per_capita, social_support, health_expectancy]]
)
st.write(f"Предсказанный уровень счастья: {predicted_score[0]:.2f}")
