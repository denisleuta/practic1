from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalize_data(data, features):
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
    return normalized