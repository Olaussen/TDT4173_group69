import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold 
from tensorflow.keras import backend as K
from sklearn.pipeline import Pipeline
from Preprocessor import Preprocessor


class Model:

    def __init__(self, data, labels):
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=0.15, random_state=1)

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
        EPOCHS = 1500
        BATCH_SIZE = 15

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('selector', VarianceThreshold()))
        estimators.append(('mlp', KerasRegressor(build_fn=generate_model, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)))
        self.model = Pipeline(estimators)
        self.model.fit(self.x_train, self.y_train)

    """
        Predicts the prices for the given data
    """
    def predict(self, data):
        return self.model.predict(data)

"""
    Returns a keras neural network model to be used for training and predictions
"""
def generate_model():
    model = Sequential()
    model.add(Dense(64, input_dim=28, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=loss, optimizer=opt)
    return model

"""
    RMSLE (Root mean squared log error) - used as a loss function for the model
"""
def loss(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return K.sqrt(msle(y_true, y_pred)) 

"""
    Calculates the RMSLE over the whole prediction set
"""
def root_mean_squared_log_error(y_true, y_pred):
    y_pred = pd.Series(y_pred)
    log_error = np.log1p(y_pred) - np.log1p(y_true)
    return np.mean(log_error ** 2) ** 0.5


"""
    Method used for saving the actual predictions to file
"""
def save_predictions(pred):
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

    model = Model(training_data, labels)
    model.fit()

    test_pred = model.predict(model.x_test)
    test_labels = model.y_test.to_numpy()
    res = pd.DataFrame([(test_labels[i], test_pred[i]) for i in range(len(test_pred))], columns=["actual", "prediction"])
    print("RMLSE: %s" % root_mean_squared_log_error(test_labels, test_pred))
    res.to_csv("split.csv")

    pred = model.predict(test_data)
    save_predictions(pred)


main()
