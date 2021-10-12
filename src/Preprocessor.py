import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from sklearn.impute import SimpleImputer

APARTMENTS_TRAIN = "../dataset/apartments_train.csv"
APARTMENTS_TEST = "../dataset/apartments_test.csv"
BUILDINGS_TRAIN = "../dataset/buildings_train.csv"
BUILDINGS_TEST = "../dataset/buildings_train.csv"

class Preprocessor:

    def __init__(self):
        self.apartments = self.load_apartments()
        self.buildings = self.load_buildings()
        self.apartments_test = self.load_apartments_test()
        self.buildings_test = self.load_buildings_test()

    def load_apartments(self):
        return pd.read_csv(APARTMENTS_TRAIN)

    def load_buildings(self):
        return pd.read_csv(BUILDINGS_TRAIN)

    def load_apartments_test(self):
        return pd.read_csv(APARTMENTS_TEST)

    def load_buildings_test(self):
        return pd.read_csv(BUILDINGS_TEST)

    def distance_from_city_center(self, lat, lon):
        city_center = {"lat": radians(55.751244), "lon": radians(37.618423)}
        radius = 6373.0 # Earth radius in kilometers
        lat = radians(lat)
        lon = radians(lon)

        delta_lat = city_center["lat"] - lat
        delta_lon = city_center["lon"] - lon

        a = sin(delta_lat / 2)**2 + cos(lat) * cos(lat) * sin(delta_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return  radius * c

    def preprocessed_buildings(self, test_set = False):
        data = self.buildings_test if test_set else self.buildings
        return data
    
    def preprocessed_apartments(self, test_set = False):
        data = self.apartments_test if test_set else self.apartments
        data = data.drop(["id"] if test_set else ["id", "price"], axis=1)
        imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        return imp.fit_transform(data)



    