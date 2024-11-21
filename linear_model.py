from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_linear_model(data):
    X = data[
        [
            "Cost of Living Index",
            "AQI Value",
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
    return model, predicted_scores
