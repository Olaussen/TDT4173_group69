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
from Preprocessor import Preprocessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
# import xgboost

class TestModel:

    def __init__(self, data, labels):
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=0.33)

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
        EPOCHS = 250
        BATCH_SIZE = 10

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('selector', VarianceThreshold()))

        #rf = RandomForestRegressor(max_depth=75, max_features="auto", min_samples_leaf=1, min_samples_split=2, n_estimators=80, verbose=2)
        #grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-4, verbose=2, scoring="neg_mean_squared_log_error")
        #estimators.append(('rfr', rf))
        #cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
        #ridge = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring="neg_mean_squared_log_error")
        #estimators.append(("ridge", ridge))
        boost = xgboost.XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.4, gamma=0, learning_rate=0.07, max_delta_step=0, max_depth=3, min_child_weight=1.5,
                                     missing=None, n_estimators=10000, nthread=-1, objective='reg:linear', reg_alpha=0.75, reg_lambda=0.45, scale_pos_weight=1, seed=42, silent=True, subsample=0.6)
        #estimators.append(('mlp', KerasRegressor(build_fn=self.generate_model, batch_size=BATCH_SIZE, epochs=EPOCHS)))
        estimators.append(("boost", boost))
        self.model = Pipeline(estimators)
        return self.model.fit(self.x_train, self.y_train)
        #print("Best params:", grid_search.best_params_)

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
            units=50, input_dim=self.x_train.shape[1], activation="relu"))
        model.add(tf.keras.layers.Dense(units=25, activation="relu"))
        model.add(tf.keras.layers.Dense(units=50, activation="relu"))
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
        y_pred = pd.Series(y_pred)
        log_error = np.log1p(y_pred) - np.log1p(y_true)
        return np.mean(log_error ** 2) ** 0.5

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
