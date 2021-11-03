from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from sklearn.impute import SimpleImputer
from pickle import dump, load

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

    def logify(self, data, label, inverse=False):
        if label in data.columns:
            data[label] = np.expm1(
                data[label]) if inverse else np.log1p(data[label])
        return data

    def district_avg(self):
        districts = {}
        for _, row in pd.concat([self.merged, self.merged_test]).iterrows():
            key = str(int(row["district"] if not np.isnan(row["district"]) else -1))
            if key in districts:
                districts[key]["lat"] += row["latitude"] if not np.isnan(row["latitude"]) else 0
                districts[key]["lon"] += row["longitude"] if not np.isnan(row["longitude"]) else 0
                districts[key]["area_kitchen"] += np.log1p(row["area_kitchen"]) if not np.isnan(row["area_kitchen"]) else 0
                districts[key]["area_living"] += np.log1p(row["area_living"]) if not np.isnan(row["area_living"]) else 0
                districts[key]["bathrooms_shared"] += row["bathrooms_shared"] if not np.isnan(row["bathrooms_shared"]) else 0
                districts[key]["bathrooms_private"] += row["bathrooms_private"] if not np.isnan(row["bathrooms_private"]) else 0
                districts[key]["ceiling"] += row["ceiling"] if not np.isnan(row["ceiling"]) else 0
                districts[key]["phones"] += row["phones"] if not np.isnan(row["phones"]) else 0
                districts[key]["constructed"] += row["constructed"] if not np.isnan(row["constructed"]) else 0
                districts[key]["amount"] += 1
            else: 
                districts[key] = {} 
                districts[key]["lat"] = row["latitude"] if not np.isnan(row["latitude"]) else 0
                districts[key]["lon"] = row["latitude"] if not np.isnan(row["latitude"]) else 0
                districts[key]["area_kitchen"] = np.log1p(row["area_kitchen"]) if not np.isnan(row["area_kitchen"]) else 0
                districts[key]["area_living"] = np.log1p(row["area_living"]) if not np.isnan(row["area_living"]) else 0
                districts[key]["bathrooms_shared"] = row["bathrooms_shared"] if not np.isnan(row["bathrooms_shared"]) else 0
                districts[key]["bathrooms_private"] = row["bathrooms_private"] if not np.isnan(row["bathrooms_private"]) else 0
                districts[key]["ceiling"] = row["ceiling"] if not np.isnan(row["ceiling"]) else 0
                districts[key]["phones"] = row["phones"] if not np.isnan(row["phones"]) else 0
                districts[key]["constructed"] = row["constructed"] if not np.isnan(row["constructed"]) else 0
                districts[key]["amount"] = 1

        for district in districts:
            districts[district]["lat"] = districts[district]["lat"] / districts[district]["amount"]
            districts[district]["lon"] = districts[district]["lon"] / districts[district]["amount"]
            districts[district]["area_kitchen"] = districts[district]["area_kitchen"] / districts[district]["amount"]
            districts[district]["area_living"] = districts[district]["area_living"] / districts[district]["amount"]
            districts[district]["bathrooms_shared"] = round(districts[district]["bathrooms_shared"] / districts[district]["amount"])
            districts[district]["bathrooms_private"] = round(districts[district]["bathrooms_private"] / districts[district]["amount"])
            if districts[district]["bathrooms_shared"] == 0 and districts[district]["bathrooms_private"] == 0:
                if districts[district]["bathrooms_private"] > districts[district]["bathrooms_shared"]:
                    districts[district]["bathrooms_private"] = 1
                else:
                    districts[district]["bathrooms_shared"] = 1
            districts[district]["ceiling"] = round(districts[district]["ceiling"] / districts[district]["amount"], 3)
            districts[district]["phones"] = round(districts[district]["phones"] / districts[district]["amount"])
            districts[district]["constructed"] = round(districts[district]["constructed"] / districts[district]["amount"])
        return districts

    def combine_area_rooms(self, data):
        data["avg_room_size"] = data["area_total"] / data["rooms"]
        return data

    def combine_baths(self, data):
        data["bathroom_amount"] = data["bathrooms_private"] + data["bathrooms_shared"]
        return data

    def relative_floor(self, data):
        data["relative_floor"] = data["stories"] * data["floor"]
        return data

    
    """def find_rich_neighboors(self, data):
        rich = 16.651093950010974
        rich_neighboors = []
        for i in range(len(data)):
            a1_lat = data["latitude"][i]
            a1_lon = data["longitude"][i]
            print("Currently row:", i)
            amount = 0
            for j in range(len(data)):
                a2_lat = data["latitude"][j]
                a2_lon = data["longitude"][j]
                if i == j or data["price"][j] < rich or self.distance(a1_lat, a1_lon, a2_lat, a2_lon) > 500:
                    continue
                amount += 1
            rich_neighboors.append(amount)

        print("Dumping rich neighboors")
        with open('neighboors.pickle', 'wb+') as file:
            dump(rich_neighboors, file)
        data["rich_neighboors"] = rich_neighboors
        return data
    """

    def find_rich_neighboors(self, data, test = False):
        path = 'neighboors_test.pickle' if test else 'neighboors_train.pickle'
        with open(path, 'rb') as file:
                rich_neighboors_loaded = load(file)

        data["rich_neighboors"] = rich_neighboors_loaded
        return data
    
    def combine_windows(self, data):
        has_windows = []
        for _, row in data.iterrows():
            if row["windows_court"] == -1 and row["windows_street"] == -1:
                has_windows.append(-1)
            elif row["windows_court"] == -1 or row["windows_street"] == -1:
                if row["windows_court"] == -1:
                    if row["windows_street"] >= 1:
                        has_windows.append(1)
                    else:
                        has_windows.append(0)
                elif row["windows_street"] == -1:
                    if row["windows_court"] >= 1:
                        has_windows.append(1)
                    else:
                        has_windows.append(0)
            else:
                if row["windows_court"] >= 1 and row["windows_street"] >= 1:
                    has_windows.append(2)
                elif row["windows_court"] >= 1 or row["windows_street"] >= 1:
                    has_windows.append(1)
                else:
                    has_windows.append(0)
        data["has_windows"] = has_windows
        return data

    def remove_NaNs(self, data):
        district_average = self.district_avg()
        for i, row in data.iterrows():
            key = str(int(row["district"] if not np.isnan(row["district"]) else -1))
            if np.isnan(row["latitude"]):
                data["latitude"][i] = district_average[key]["lat"]
            if np.isnan(row["longitude"]):
                data["longitude"][i] = district_average[key]["lon"]
            if np.isnan(row["area_kitchen"]):
                data["area_kitchen"][i] = district_average[key]["area_kitchen"]
            if np.isnan(row["area_living"]):
                data["area_living"][i] = district_average[key]["area_living"]
            if np.isnan(row["bathrooms_shared"]):
                data["bathrooms_shared"][i] = district_average[key]["bathrooms_shared"]
            if np.isnan(row["bathrooms_private"]):
                data["bathrooms_private"][i] = district_average[key]["bathrooms_private"]
            if np.isnan(row["ceiling"]):
                data["ceiling"][i] = district_average[key]["ceiling"]
            if np.isnan(row["phones"]):
                data["phones"][i] = district_average[key]["phones"]
            if np.isnan(row["constructed"]):
                data["constructed"][i] = district_average[key]["constructed"]
        data['distance_center'] = [self.distance(
            data["latitude"][i], data["longitude"][i]) for i in range(len(data["latitude"]))]
        data["seller"] = data["seller"].fillna(-1)
        data["windows_court"] = data["windows_court"].fillna(-1)
        data["windows_street"] = data["windows_street"].fillna(-1)
        data["balconies"] = data["balconies"].fillna(-1)
        data["loggias"] = data["loggias"].fillna(-1)
        data["condition"] = data["condition"].fillna(-1)
        data["new"] = data["new"].fillna(-1)
        data["district"] = data["district"].fillna(-1)
        data["material"] = data["material"].fillna(-1)
        data["elevator_without"] = data["elevator_without"].fillna(-1)
        data["elevator_passenger"] = data["elevator_passenger"].fillna(
            -1)
        data["elevator_service"] = data["elevator_service"].fillna(-1)
        data["parking"] = data["parking"].fillna(-1)
        data["garbage_chute"] = data["garbage_chute"].fillna(-1)
        data["heating"] = data["heating"].fillna(-1)
        return data

    def general_removal(self, data):
        data = self.remove_labels(
            data, ["layout", "street", "address", "id", "building_id"])
        return data

    def remove_redundant_features(self, data):
        data = self.remove_labels(data, ["balconies", "parking", "loggias", "condition", "heating", "ceiling", "windows_street", "windows_court", "elevator_passenger", "phones", "new", "elevator_service", "material"])
        return data

    def distance(self, lat, lon, lat_to = 55.753649, lon_to = 37.621067):
        radius = 6373000.0  # Earth radius in meters
        lat = radians(lat)
        lon = radians(lon)

        delta_lat = radians(lat_to) - lat
        delta_lon = radians(lon_to) - lon

        a = sin(delta_lat / 2)**2 + cos(lat) * cos(lat) * sin(delta_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return radius * c

    def remove_labels(self, data, labels):
        for label in labels:
            if label in data.columns:
                data = data.drop([label], axis=1)
        return data

    def merge_apartment_building(self):
        result = pd.merge(self.apartments, self.buildings.set_index(
            'id'), how='left', left_on='building_id', right_index=True)
        result.to_csv("../dataset/merged.csv", index=False)
        return result

    def preprocess(self, data, test=False):
        data = self.logify(data.copy(), "price")
        data = self.logify(data, "area_total")
        data = self.logify(data, "area_living")
        data = self.logify(data, "area_kitchen")
        data = self.remove_NaNs(data)
        data = self.general_removal(data)
        #data = self.find_rich_neighboors(data, test=test)
        data = self.remove_redundant_features(data)
        #data = self.combine_area_rooms(data)
        data = self.combine_baths(data)
        #data = self.combine_windows(data)
        data = self.relative_floor(data)
        data = self.remove_labels(
        data, labels=["bathrooms_private", "bathrooms_shared", "latitude", "longitude"])
        return data

def main():
    p = Preprocessor()
    p.preprocess(p.merged)
    
#main()
