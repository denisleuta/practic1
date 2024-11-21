import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

# Функция для отображения фона
def show_background(image_path):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Гистограмма уровня счастья
def plot_happiness_histogram(data):
    plt.figure(figsize=(10, 6))
    plt.hist(data["Score"], bins=20, color='skyblue', edgecolor='black')
    plt.title("Распределение уровня счастья")
    plt.xlabel("Уровень счастья")
    plt.ylabel("Частота")
    plt.show()

# Карта Кохонена
def plot_som_map(som, data, clusters):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Карта расстояний между нейронами")
    ax.imshow(som.distance_map().T, cmap="coolwarm", interpolation="nearest")
    for i, (x, y) in enumerate(clusters):
        ax.text(x, y, data["Country"].iloc[i], fontsize=8, ha="center", va="center")
    plt.show()

# Генерация карты с использованием Plotly
def plot_map(data, filtered_data):
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
    fig_map.show()