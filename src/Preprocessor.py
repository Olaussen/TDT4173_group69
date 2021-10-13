import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from sklearn.impute import SimpleImputer

APARTMENTS_TRAIN = "../dataset/apartments_train.csv"
APARTMENTS_TEST = "../dataset/apartments_test.csv"
BUILDINGS_TRAIN = "../dataset/buildings_train.csv"
BUILDINGS_TEST = "../dataset/buildings_test.csv"
MERGED_TRAIN = "../dataset/merged.csv"
MERGED_TEST = "../dataset/merged_test.csv"


class Preprocessor:

    def __init__(self):
        self.apartments = self.load_apartments()
        self.buildings = self.load_buildings()
        self.apartments_test = self.load_apartments_test()
        self.buildings_test = self.load_buildings_test()
        self.merged = self.load_merged()
        self.merged_test = self.load_merged_test()

    def load_apartments(self):
        return pd.read_csv(APARTMENTS_TRAIN)

    def load_buildings(self):
        return pd.read_csv(BUILDINGS_TRAIN)

    def load_apartments_test(self):
        return pd.read_csv(APARTMENTS_TEST)

    def load_buildings_test(self):
        return pd.read_csv(BUILDINGS_TEST)
    
    def load_merged(self):
        return pd.read_csv(MERGED_TRAIN)
    
    def load_merged_test(self):
        return pd.read_csv(MERGED_TEST)

    def distance_from_city_center(self, lat, lon):
        city_center = {"lat": radians(55.751244), "lon": radians(37.618423)}
        radius = 6373.0  # Earth radius in kilometers
        lat = radians(lat)
        lon = radians(lon)

        delta_lat = city_center["lat"] - lat
        delta_lon = city_center["lon"] - lon

        a = sin(delta_lat / 2)**2 + cos(lat) * cos(lat) * sin(delta_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return radius * c

    def remove_labels(self, data, labels):
        for label in labels:
            if label in data.columns:
                data = data.drop([label], axis=1)
        return data

    def merge_apartment_building(self):
        result = pd.merge(self.apartments, self.buildings.set_index('id'), how='left', left_on='building_id', right_index=True)
        result.to_csv("../dataset/merged.csv", index = False)
        return result

    def preprocess(self, data, impute = False):
        result = self.remove_labels(data, ["id", "building_id", "street", "address", "seller", "price"])
        distances = []

        for _, row in result.iterrows():
            distances.append(self.distance_from_city_center(row["latitude"], row["longitude"]))
        result["distance"] = distances

        if impute:
            return self.impute(result)
        return result

    def impute(self, data):
        imp = SimpleImputer(missing_values=np.nan, strategy="constant")
        return imp.fit_transform(data)
