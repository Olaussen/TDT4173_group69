import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.normalization import BatchNormalization
from Preprocessor import Preprocessor
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class Model:

    def __init__(self, data, labels):
        self.model = None
        self.score = None
        self.loss = None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=0.2, random_state=0)

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
        EPOCHS = 25
        BATCH_SIZE = 5
        FOLDS = 10

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(build_fn=generate_model, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)))
        self.model = Pipeline(estimators)
        self.model.fit(self.x_train, self.y_train)
        test_pred = self.predict(self.x_test) # Test partition prediction

        kfold = KFold(n_splits=FOLDS)
        results = cross_val_score(self.model, self.x_test, test_pred, cv=kfold, verbose=1, n_jobs=-1)

        print("Results: %.4f (%.4f) RMSLE" % (results.mean(), results.std()))
        print("Root mean squared log error on test partition: %s" % root_mean_squared_log_error(self.y_test, test_pred))

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
    model.add(Dense(32, input_dim=29, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss=loss, optimizer='adam')
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

    merged = preprocessor.merged
    merged_test = preprocessor.merged_test

    training_data = preprocessor.preprocess(merged, impute=True)
    labels = preprocessor.apartments["price"]

    test_data = preprocessor.preprocess(merged_test, impute=True)

    model = Model(training_data, labels)
    model.fit()
    pred = model.predict(test_data)
    save_predictions(pred)


main()
