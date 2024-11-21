from minisom import MiniSom
from .normalization import normalize_data


def train_som(data, features, som_size=(10, 10)):
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
    som.train_random(data=normalized_data, num_iteration=1000)

    # Присвоение кластеров
    clusters = [som.winner(row) for row in normalized_data]
    return som, clusters
