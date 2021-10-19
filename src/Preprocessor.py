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

    def preprocess(self, data):
        data["seller"]=data["seller"].fillna(4)
        data["area_kitchen"]=data["area_kitchen"].fillna(data["area_kitchen"].mean())
        data["area_living"]=data["area_living"].fillna(data["area_living"].mean())
        data.drop('layout',1,inplace=True)
        data["ceiling"]=data["ceiling"].fillna(data["ceiling"].mean())
        data["bathrooms_shared"]=data["bathrooms_shared"].fillna(data["bathrooms_shared"].median())
        data["bathrooms_private"]=data["bathrooms_private"].fillna(data["bathrooms_private"].median())
        data["windows_court"]=data["windows_court"].fillna(2)
        data["windows_street"]=data["windows_street"].fillna(2)
        data["balconies"]=data["balconies"].fillna(0)
        data["loggias"]=data["loggias"].fillna(0)
        data["condition"]=data["condition"].fillna(4)
        data["phones"]=data["phones"].fillna(data["phones"].median())
        data["new"]=data["new"].fillna(2)
        data["longitude"]=data["longitude"].fillna(data["longitude"].mean())
        data["latitude"]=data["latitude"].fillna(data["latitude"].mean())
        data['distance_center']=np.linalg.norm(np.array([data["latitude"].to_numpy(),data["longitude"].to_numpy()]).T-np.array([55.7521,37.6211]),axis=1)
        data.drop('latitude',1,inplace=True)
        data.drop('longitude',1,inplace=True)
        data["district"]=data["district"].fillna(0)
        data["constructed"]=data["constructed"].fillna(data["constructed"].median())
        data["material"]=data["material"].fillna(7)
        data["elevator_without"]=data["elevator_without"].fillna(2)
        data["elevator_passenger"]=data["elevator_passenger"].fillna(2)
        data["elevator_service"]=data["elevator_service"].fillna(2)
        data["parking"]=data["parking"].fillna(3)
        data["garbage_chute"]=data["garbage_chute"].fillna(2)
        data["heating"]=data["heating"].fillna(4)
        data.drop('street',1,inplace=True)
        data.drop('address',1,inplace=True)
        data.drop('id',1,inplace=True)
        return data
