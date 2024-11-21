import pandas as pd

# Загрузка данных
def load_data():
    happiness = pd.read_csv("data/2019.csv")
    cost_of_living = pd.read_csv("data/Cost_of_Living_Index_by_Country_2024.csv")
    pollution = pd.read_csv("data/global_air_pollution_dataset.csv")
    coordinates = pd.read_csv("data/World Cities Nearest Stations.csv")

    # Предобработка данных
    happiness.rename(columns={"Country or region": "Country"}, inplace=True)
    coordinates.rename(
        columns={"country": "Country", "lat": "latitude", "lng": "longitude"},
        inplace=True,
    )

    # Удаление дубликатов
    happiness = happiness.drop_duplicates(subset="Country")
    cost_of_living = cost_of_living.drop_duplicates(subset="Country")
    pollution = pollution.drop_duplicates(subset="Country")
    coordinates = coordinates.drop_duplicates(subset="Country")

    # Объединение данных
    data = (
        happiness.merge(cost_of_living, on="Country", how="inner")
        .merge(pollution, on="Country", how="inner")
        .merge(coordinates, on="Country", how="left")  # Добавляем координаты
    )

    # Удаление строк с пропущенными значениями
    data.dropna(inplace=True)
    return data
