import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns 
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class Model:

    def __init__(self):
        self.model = None
        self.score = None
        self.loss = None

    def predict_linearly(self, m, b, data):
        predictions = []
        for _, row in data.iterrows():
            price = m * row["area_total"] + b
            if price < 0:
                price = 0
            predictions.append((int(row["id"]), price))
        return pd.DataFrame(predictions, columns=["id", "price"])

    def fit(self, data, labels):
        """model = LinearRegression()
        model.fit(x_train, y_train)
        self.score = model.score(x_test, y_test)
        self.model = model
        return model"""
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=10, batch_size=5, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=10)
        results = cross_val_score(pipeline, data, labels, cv=kfold)
        print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
        return estimators


    def predict(self, data):
        pass

    def root_mean_squared_log_error(self, y_true, y_pred):
        y_pred = pd.Series(y_pred)
        assert (y_true >= 0).all()
        assert (y_pred >= 0).all()
        log_error = np.log1p(y_pred) - np.log1p(y_true)
        return np.mean(log_error ** 2) ** 0.5

    def save_predictions(self, pred):
        zipped = [(23285+i, pred[i]) for i in range(len(pred))]
        result=pd.DataFrame(zipped, columns = ["id", "price_prediction"])
        result.to_csv("../results/predictions.csv", index = False)
        return result

def larger_model():
        model = Sequential()
        model.add(Dense(27, input_dim=27, kernel_initializer='normal', activation='relu'))
        model.add(Dense(6, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
        return model