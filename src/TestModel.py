import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras import backend as K
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from Preprocessor import Preprocessor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
import xgboost
import lightgbm as lgbm
from flaml import AutoML
import json


def root_mean_squared_log_error(y_true, y_pred):
    return mean_squared_log_error(y_true, y_pred) ** 0.5


def custom_metric(X_test, y_test, estimator, labels, X_train, y_train,
                  weight_test=None, weight_train=None, config=None, groups_test=None, groups_train=None):
    y_pred = estimator.predict(X_test)
    test_loss = root_mean_squared_log_error(y_test, y_pred)
    y_pred = estimator.predict(X_train)
    train_loss = root_mean_squared_log_error(y_train, y_pred)
    alpha = 0.5
    return test_loss * (1 + alpha) - alpha * train_loss, {}


class TestModel:

    def __init__(self, x_train, y_train):
        self.model = None
        self.x_train = x_train
        self.y_train = y_train

    """
        This function is used just for showing that trying to predict the price based on the total area alone will
        not be good enough for an accurate prediction
    """

    def predict_linearly(self, m, b, data):
        predictions = []
        for _, row in data.iterrows():
            price = m * row["area_total"] + b
            if price < 0:
                price = 0
            predictions.append((int(row["id"]), price))
        return pd.DataFrame(predictions, columns=["id", "price"])

    """
        Main function for training the model. Will use the dataset partition selected in the constructor to
        fit a KerasRegressor model. The trained model is bound to self.model.
    """

    def fit(self):
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('selector', VarianceThreshold()))
        # Set paramters for Grid Search
        # param_grid = {'n_estimators': [200, 300, 400, 500, 600],
        # 'max_features': [0.1, 0.3, 0.6]
        # }
        # Initialise the random forest model
        #rf = RandomForestRegressor(n_jobs=-1, random_state=0, bootstrap=True)

        # Initialise Gridsearch CV with 5 fold corssvalidation and neggative root_mean_squared_error
        # tuned_rf = GridSearchCV(
        # estimator=rf, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, verbose=2)

        # rf = RandomForestRegressor(bootstrap=True,
        # max_depth=40,
        # max_features='auto',
        # min_samples_leaf=2,
        # min_samples_split=5,
        # n_estimators=100, verbose=2)
        #estimators.append(('rfr', rf))
        #gr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.5)
        #estimators.append(("gradient", gr))
        params = {'n_estimators': 1168, 'num_leaves': 118, 'min_child_samples': 7, 'learning_rate': 0.08071528250529435,
                  'log_max_bin': 10, 'colsample_bytree': 0.662586923419352, 'reg_alpha': 0.0031039225181313645, 'reg_lambda': 0.05061507015311157}
        boost = xgboost.XGBRegressor(**params)
        estimators.append(("boost", boost))
        self.model = Pipeline(estimators)
        #self.model = rf
        return self.model.fit(self.x_train, self.y_train)
        #print("Best params:", grid_search.best_params_)

    def keras_mlp_model(self, epochs=100, batch_size=10, verbose=0):
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('selector', VarianceThreshold()))
        estimators.append(('mlp', KerasRegressor(
            build_fn=self.generate_model, epochs=epochs, batch_size=batch_size, verbose=verbose)))
        return Pipeline(estimators)

    def randomforest_model(self):
        rf = RandomForestRegressor()
        return rf

    def xgboost_model(self, params = {}):
        boost = xgboost.XGBRegressor(**params)
        return boost

    def lgbm_model(self, params = {}):
        lightgbm = lgbm.LGBMRegressor(**params)
        return lightgbm

    def start_rf_search(self, params, load=False):
        if load:
            with open("rf_best.json", "r+") as file:
                best = json.load(file)
            return RandomForestRegressor(**best)
        else:
            rf = RandomForestRegressor()
            finished = GridSearchCV(
                estimator=rf, param_grid=params, cv=3, verbose=2, n_jobs=-1)
            return finished

    def start_xgboost_search(self, params, load=False):
        if load:
            with open("boost_best.json", "r+") as file:
                best = json.load(file)
            return xgboost.XGBRegressor(**best)
        else:
            boost = xgboost.XGBRegressor()
            finished = GridSearchCV(
                estimator=boost, param_grid=params, cv=3, verbose=2, n_jobs=-1)
            return finished

    def start_lgbm_search(self, params, load=False):
        if load:
            with open("lgbm_best.json", "r+") as file:
                best = json.load(file)
            return lgbm.LGBMRegressor(**best)
        else:
            lg = lgbm.LGBMRegressor()
            finished = GridSearchCV(
                estimator=lg, param_grid=params, cv=3, verbose=2, n_jobs=-1)
            return finished

    def autoMLfit(self, x_train, y_train, estimator_list=["xgboost", "lgbm", "rf"], time=10, metric=custom_metric, ensemble=False):
        automl_settings = {
            "time_budget": time,  # in seconds
            "metric": metric,
            "task": 'regression',
            "log_file_name": "lmaoxd.log",
            "estimator_list": estimator_list,
            "ensemble": ensemble,
        }
        automl = AutoML()
        self.model = automl
        return self.model.fit(x_train, y_train, **automl_settings)

    def autoMLpredict(self, x_test):
        return self.model.predict(x_test)

    def autoML_print_best_model(self):
        print("best model", self.model.best_estimator)
        print("configs", self.model.best_config)

    """
        Predicts the prices for the given data
    """

    def predict(self, data):
        return self.model.predict(data)

    """
        Returns a keras neural network model to be used for training and predictions
    """

    def generate_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(
            units=64, activation="relu"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        model.add(tf.keras.layers.Dense(units=1))
        model.compile(optimizer='adam', loss=self.loss)
        return model

    """
        RMSLE (Root mean squared log error) - used as a loss function for the model
    """

    def loss(self, y_true, y_pred):
        msle = tf.keras.losses.MeanSquaredLogarithmicError()
        return K.sqrt(msle(y_true, y_pred))

    """
        Calculates the RMSLE over the whole prediction set
    """

    def root_mean_squared_log_error(self, y_true, y_pred):
        return mean_squared_log_error(y_true, y_pred) ** 0.5

    """
        Method used for saving the actual predictions to file
    """

    def save_predictions(self, pred):
        zipped = [(23285+i, pred[i]) for i in range(len(pred))]
        result = pd.DataFrame(zipped, columns=["id", "price_prediction"])
        result.to_csv("../results/predictions.csv", index=False)
        return result


def main():
    preprocessor = Preprocessor()

    labels = preprocessor.apartments["price"]
    merged = preprocessor.merged
    merged_test = preprocessor.merged_test

    training_data = preprocessor.preprocess(merged)
    training_data.drop("price", 1, inplace=True)
    test_data = preprocessor.preprocess(merged_test)

    model = TestModel(training_data, labels)
    model.fit()

    test_pred = model.predict(model.x_test)
    test_labels = model.y_test.to_numpy()

    fig, ax = plt.subplots()
    ax.scatter(test_labels, test_pred)
    ax.plot([test_labels.min(), test_labels.max()], [
            test_labels.min(), test_labels.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    res = pd.DataFrame([(test_labels[i], test_pred[i]) for i in range(
        len(test_pred))], columns=["actual", "prediction"])
    print("RMLSE: %s" % model.root_mean_squared_log_error(test_labels, test_pred))
    res.to_csv("split.csv", index=False)

    pred = model.predict(test_data)
    model.save_predictions(pred)


# main()
