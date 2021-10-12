import numpy as np
import pandas as pd

class Model:

    def __init__(self):
        pass

    def predict_linearly(self, m, b, data):
        predictions = []
        for _, row in data.iterrows():
            price = m * row["area_total"] + b
            if price < 0:
                price = 0
            predictions.append((int(row["id"]), price))
        return pd.DataFrame(predictions, columns=["id", "price"])

    def fit(self, data):
        print("Fitting")
      

    def predict(self, data):
        pass

    def root_mean_squared_log_error(self, y_true, y_pred):
        assert (y_true >= 0).all() 
        assert (y_pred >= 0).all()
        log_error = np.log1p(y_pred) - np.log1p(y_true)
        return np.mean(log_error ** 2) ** 0.5

    def save_predictions(self, pred):
        result = pd.DataFrame(pred, columns=["id", "price_prediction"])
        result.to_csv("../results/predictions.csv")
