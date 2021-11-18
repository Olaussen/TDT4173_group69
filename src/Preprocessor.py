from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from pickle import dump, load
from scipy import stats

from pandas.core.base import DataError
from sklearn.preprocessing import OneHotEncoder
pd.options.mode.chained_assignment = None  # default='warn'

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
        self.district_avg_dict = None

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
    
    def log2ify(self, data, label, inverse=False):
        if label in data.columns:
            data[label] = np.exp2(
                data[label]) if inverse else np.log2(data[label])
        return data
    
    def squareify(self, data, label, inverse=False):
        if label in data.columns:
            data[label] = np.exp2(
                data[label]) if inverse else np.sqrt(data[label])
        return data
    
    def skew_fix(self,data,label, inverse=False):
        if label in data.columns:
            data[label] = stats.boxcox(data[label])
        return data
    
    def combine_floor_stories(self,data):
        data["floor_stories"] = data["floor"]*data["stories"]
        return data
    
    def floor_fraction(self,data):
        data["floor_fraction"] = data["floor"]/data["stories"]
        return data
    
    def combine_new_constructed_distance(self,data):
        data["scaled_constructed"] = data["constructed"]*data["new"]
        for i in data.index:
            if data.at[i,"scaled_constructed"] == 0:
                data.at[i,"scaled_constructed"] = data.at[i,"constructed"]*0.70
        return data
    
    def combine_district_city_center(self,data):
        data["district_distance"] = data["district"]*data["distance_center"]
        return data
    
    def combine_area_total_city_center(self,data):
        data["area_total_distance"] = data["area_total"]/np.log1p(data["distance_center"])
        return data
    
    def combine_district_bathroom_amount(self,data):
        data["district_bath_amount"] = data["district"]*data["bathroom_amount"]
        return data
    
    def square_diff(self,data):
        data["square_diff"] = data["area_total"]-(data["area_kitchen"])
        return data
    
    def rich_square(self,data):
        data["richy_square_score"] = 10000
        for i in data.index:
            # richy rich square
            if ((55.7<=data.at[i,"latitude"]<=55.8) and (37.5<=data.at[i,"longitude"]<=37.68)):
                data.at[i,"richy_square_score"] = 1
            elif((55.7<=data.at[i,"latitude"]<=55.8) and (37.39<=data.at[i,"longitude"]<=37.49)):
                data["richy_square_score"] = 2
            elif((55.64<=data.at[i,"latitude"]<=55.7) and (37.42<=data.at[i,"longitude"]<=37.585)):
                data.at[i,"richy_square_score"] = 7
            elif((55.65<=data.at[i,"latitude"]<=56.0) and (37.2<=data.at[i,"longitude"]<=37.78)):
                data.at[i,"richy_square_score"] = 1000
        return data
        
    def closest_hospital(self, data):
        with open("hospitals.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            hospitals = []
        for _, row in data.iterrows():
            closest = float("inf")
            for hospital in coords:
                distance = self.distance(row["latitude"], row["longitude"], hospital[0], hospital[1])
                if distance < closest:
                    closest = distance
            hospitals.append(closest)
        data["closest_hospital"] = hospitals
        return data
    
    def closest_park(self, data):
        with open("parks.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            parks = []
        for _, row in data.iterrows():
            closest = float("inf")
            for park in coords:
                distance = self.distance(row["latitude"], row["longitude"], park[0], park[1])
                if distance < closest:
                    closest = distance
            parks.append(closest)
        data["closest_park"] = parks
        return data
    
    def closest_uni(self, data):
        with open("uni.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            unis = []
        for _, row in data.iterrows():
            closest = float("inf")
            for uni in coords:
                distance = self.distance(row["latitude"], row["longitude"], uni[0], uni[1])
                if distance < closest:
                    closest = distance
            unis.append(closest)
        data["closest_uni"] = unis
        return data

    def area_score(self,data):
        data["area_score"] = data["area_total"]/((data["district"]+2)**2)
        return data
    
    def get_closest_district(self, data, non_district):
        copy = data.copy()[data["district"].notna()]
        for i, row in non_district.iterrows():
            best = None
            closest = float("inf")
            for _, check in copy.iterrows():
                distance = self.distance(
                    row["latitude"], row["longitude"], check["latitude"], check["longitude"])
                if distance < 250:
                    data["district"][i] = check["district"]
                    break
                if distance < closest:
                    closest = distance
                    best = row["district"]
            non_district["district"][i] = best
        return non_district

    def split_categorical_features(self, data, columns):
        i = 0
        for field in columns:
            df1 = pd.get_dummies(
                data[field], drop_first=True, prefix=columns[i])
            data.drop([field], axis=1, inplace=True)

            if i == 0:
                df_final = df1.copy()
            else:
                df_final = pd.concat([df_final, df1], axis=1)
            i = i + 1

        df_final = pd.concat([data, df_final], axis=1)
        return df_final

    def district_avg(self, data):
        districts = {}
        for _, row in data.iterrows():
            key = str(
                int(row["district"] if not np.isnan(row["district"]) else -1))
            if key in districts:
                districts[key]["lat"] += row["latitude"] if not np.isnan(
                    row["latitude"]) else 0
                districts[key]["lon"] += row["longitude"] if not np.isnan(
                    row["longitude"]) else 0
                districts[key]["area_kitchen"] += np.log1p(
                    row["area_kitchen"]) if not np.isnan(row["area_kitchen"]) else 0
                districts[key]["area_living"] += np.log1p(
                    row["area_living"]) if not np.isnan(row["area_living"]) else 0
                districts[key]["bathrooms_shared"] += row["bathrooms_shared"] if not np.isnan(
                    row["bathrooms_shared"]) else 0
                districts[key]["bathrooms_private"] += row["bathrooms_private"] if not np.isnan(
                    row["bathrooms_private"]) else 0
                districts[key]["phones"] += row["phones"] if not np.isnan(
                    row["phones"]) else 0
                districts[key]["constructed"] += row["constructed"] if not np.isnan(
                    row["constructed"]) else 0
                districts[key]["amount"] += 1
            else:
                districts[key] = {}
                districts[key]["lat"] = row["latitude"] if not np.isnan(
                    row["latitude"]) else 0
                districts[key]["lon"] = row["latitude"] if not np.isnan(
                    row["latitude"]) else 0
                districts[key]["area_kitchen"] = np.log1p(
                    row["area_kitchen"]) if not np.isnan(row["area_kitchen"]) else 0
                districts[key]["area_living"] = np.log1p(
                    row["area_living"]) if not np.isnan(row["area_living"]) else 0
                districts[key]["bathrooms_shared"] = row["bathrooms_shared"] if not np.isnan(
                    row["bathrooms_shared"]) else 0
                districts[key]["bathrooms_private"] = row["bathrooms_private"] if not np.isnan(
                    row["bathrooms_private"]) else 0
                districts[key]["phones"] = row["phones"] if not np.isnan(
                    row["phones"]) else 0
                districts[key]["constructed"] = row["constructed"] if not np.isnan(
                    row["constructed"]) else 0
                districts[key]["amount"] = 1

        for district in districts:
            districts[district]["lat"] = districts[district]["lat"] / \
                districts[district]["amount"]
            districts[district]["lon"] = districts[district]["lon"] / \
                districts[district]["amount"]
            districts[district]["area_kitchen"] = districts[district]["area_kitchen"] / \
                districts[district]["amount"]
            districts[district]["area_living"] = districts[district]["area_living"] / \
                districts[district]["amount"]
            districts[district]["bathrooms_shared"] = round(
                districts[district]["bathrooms_shared"] / districts[district]["amount"])
            districts[district]["bathrooms_private"] = round(
                districts[district]["bathrooms_private"] / districts[district]["amount"])
            districts[district]["phones"] = round(
                districts[district]["phones"] / districts[district]["amount"])
            districts[district]["constructed"] = round(
                districts[district]["constructed"] / districts[district]["amount"])
        return districts

    def combine_area_rooms(self, data):
        data["avg_room_size"] = data["area_total"] / (data["rooms"]**3)
        data["living_fraction"] = data["area_living"] / data["area_total"]
        data["kitchen_fraction"] = data["area_kitchen"] / data["area_total"]
        return data

    def combine_baths(self, data):
        data["bathroom_amount"] = data["bathrooms_private"]+data["bathrooms_shared"]
        return data
    
    def bathroom_fraction(self, data):
        data["bathroom_fraction"] = data["bathroom_amount"]/data["area_total"]
        return data

    def relative_floor(self, data):
        data["relative_floor"] = data["stories"] * data["floor"]
        return data
    
    def lat_long_fraction(self, data):
        data["lat_lon_frac"] = data["longitude"]-data["latitude"]
        return data
    
    def apartment_score(self, data):
        data["apartment_score"] = (data["has_elevator"]+data["windows_street"]+data["windows_court"]+data["parking"]+data["heating"]+data["phones"]+data["bathroom_amount"]+(data["floor"]/data["stories"]))
        return data
    
    def distance_luxury_village(self,data):
        data["distance_luxury_village"] = [self.distance(data["latitude"][i], data["longitude"][i],55.737886288945795,37.25939226060453) for i in range(len(data["latitude"]))]
        return data
    
    def inside_golden_mile(self,data):
        data["inside_golden_mile"] = 0
        center = (55.73953654149639, 37.60149628258151)
        # km
        radius = 0.65
        for i in data.index:
            distance_from_center = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],center[0],center[1])
            if(distance_from_center<= radius):
                data.at[i,"inside_golden_mile"] = 1
        return data
    
    def distance_from_golden_mile(self,data):
        data["distance_golden_mile"] = [self.distance(data["latitude"][i], data["longitude"][i],55.7391511539523, 37.59601968052924) for i in range(len(data["latitude"]))]
        return data
    
    def inside_khamovniki(self,data):
        data["inside_khamovniki"] = 0
        centers = [(55.72080955799734, 37.56532305228938),(55.733461209237284, 37.57853367452259),(55.74283408846054, 37.60129837589054),(55.740730168657976, 37.57700470204264)]
        rad1 = 1.45
        rad2 = 0.6
        for i in data.index:
            distance_from_center1 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[0][0],centers[0][1])
            distance_from_center2 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[1][0],centers[1][1])
            distance_from_center3 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[2][0],centers[2][1])
            distance_from_center4 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[3][0],centers[3][1])
            if((distance_from_center1<= rad1)or(distance_from_center2<= rad2)or(distance_from_center3<= rad2)or(distance_from_center4<= rad2)):
                data.at[i,"inside_khamovniki"] = 1
        return data
    
    def distance_from_khamovniki_center(self,data):
        data["distance_khamovniki_center"] = [self.distance(data["latitude"][i], data["longitude"][i],55.733829823959425, 37.57598361150507) for i in range(len(data["latitude"]))]
        return data
    
    def inside_khamovniki_and_yakimanka(self,data):
        data["inside_khamovniki_yakimanka"] = 0
        centers = [(55.722842003327585, 37.574932334725695),(55.733313673521124, 37.58786146165378)]
        rad1 = 1.4
        rad2 = 2.0
        for i in data.index:
            distance_from_center1 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[0][0],centers[0][1])
            distance_from_center2 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[1][0],centers[1][1])
            if((distance_from_center1<= rad1)or(distance_from_center2<= rad2)):
                data.at[i,"inside_khamovniki_yakimanka"] = 1
        return data
    
    def inside_yakimanka(self,data):
        data["inside_yakimanka"] = 0
        centers = [(55.74163081394598, 37.61627821363091),(55.73627260408775, 37.61346612143006),(55.73077227529825, 37.60997299730604),(55.72605357100986, 37.5996889100099)]
        rad1 = 0.6
        rad2 = 0.65
        rad3 = 0.78
        for i in data.index:
            distance_from_center1 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[0][0],centers[0][1])
            distance_from_center2 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[1][0],centers[1][1])
            distance_from_center3 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[2][0],centers[2][1])
            distance_from_center4 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[3][0],centers[3][1])
            if((distance_from_center1<= rad1)or(distance_from_center2<= rad2)or(distance_from_center3<= rad2)or(distance_from_center4<= rad3)):
                data.at[i,"inside_yakimanka"] = 1
        return data
    
    def inside_arbat(self,data):
        data["inside_arbat"] = 0
        centers = [(55.75275363343967, 37.607045328488844),(55.74527362997916, 37.58460600370206),(55.75168130861189, 37.59244539997004),(55.75059904094849, 37.583070554322056)]
        rad1 = 0.31
        rad2 = 0.12
        rad3 = 0.53
        rad4 = 0.6
        for i in data.index:
            distance_from_center1 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[0][0],centers[0][1])
            distance_from_center2 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[1][0],centers[1][1])
            distance_from_center3 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[2][0],centers[2][1])
            distance_from_center4 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[3][0],centers[3][1])
            if((distance_from_center4<= rad4)or(distance_from_center3<= rad3)or(distance_from_center1<= rad1)or(distance_from_center2<= rad2)):
                data.at[i,"inside_arbat"] = 1
        return data
    
    def inside_tverskoy(self,data):
        data["inside_tverskoy"] = 0
        centers = [(55.788417781850306, 37.593312026018964),(55.77511666189357, 37.603314054940284),(55.76831412199218, 37.61003700393288),(55.76203646189537, 37.61389958917979),(55.75384950055093, 37.62029284325331)]
        rad1 = 0.42
        rad2 = 1
        rad3 = 0.73
        rad4 = 0.4
        rad5 = 0.83
        for i in data.index:
            distance_from_center1 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[0][0],centers[0][1])
            distance_from_center2 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[1][0],centers[1][1])
            distance_from_center3 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[2][0],centers[2][1])
            distance_from_center4 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[3][0],centers[3][1])
            distance_from_center5 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[4][0],centers[4][1])
            if((distance_from_center1<= rad1)or(distance_from_center2<= rad2)or(distance_from_center3<= rad3)or(distance_from_center4<= rad4)or(distance_from_center5<= rad5)):
                data.at[i,"inside_tverskoy"] = 1
        return data
    
    def inside_zamoskvorechye(self,data):
        data["inside_zamoskvorechye"] = 0
        centers = [(55.724772756233676, 37.62366383340057),(55.731251076773965, 37.63653395506052),(55.73835792252437, 37.635563409780225),(55.74466143972234, 37.633643796367785)]
        rad1 = 0.45
        rad2 = 0.81
        rad3 = 0.7
        rad4 = 0.53
        for i in data.index:
            distance_from_center1 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[0][0],centers[0][1])
            distance_from_center2 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[1][0],centers[1][1])
            distance_from_center3 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[2][0],centers[2][1])
            distance_from_center4 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[3][0],centers[3][1])
            if((distance_from_center1<= rad1)or(distance_from_center2<= rad2)or(distance_from_center3<= rad3)or(distance_from_center4<= rad4)):
                data.at[i,"inside_zamoskvorechye"] = 1
        return data
    
    def inside_presnensky(self,data):
        data["inside_presnensky"] = 0
        centers = [(55.75391845697493, 37.53422062903805),(55.76003533601604, 37.54938877500197),(55.76473880878593, 37.56463365341768),(55.76446917930647, 37.581678475143406),(55.75963515297629, 37.5976562520681)]
        rad1 = 0.9
        rad2 = 1.12
        rad3 = 1.15
        rad4 = 1.1
        rad5 = 0.52
        for i in data.index:
            distance_from_center1 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[0][0],centers[0][1])
            distance_from_center2 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[1][0],centers[1][1])
            distance_from_center3 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[2][0],centers[2][1])
            distance_from_center4 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[3][0],centers[3][1])
            distance_from_center5 = self.distance(data.at[i,"latitude"],data.at[i,"longitude"],centers[4][0],centers[4][1])
            if((distance_from_center1<= rad1)or(distance_from_center2<= rad2)or(distance_from_center3<= rad3)or(distance_from_center4<= rad4)or(distance_from_center5<= rad5)):
                data.at[i,"inside_presnensky"] = 1
        return data
    
    def distance_to_closest_powerplant(self,data):
        with open("power_plants.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            plants = []
        for _, row in data.iterrows():
            closest = float("inf")
            for plant in coords:
                distance = self.distance(row["latitude"], row["longitude"], plant[0], plant[1])
                if distance < closest:
                    closest = distance
            plants.append(closest)
        data["distance_closest_powerplant"] = plants
        return data
    
    def distance_to_closest_museum(self,data):
        with open("museums.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            museums = []
        for _, row in data.iterrows():
            closest = float("inf")
            for museum in coords:
                distance = self.distance(row["latitude"], row["longitude"], museum[0], museum[1])
                if distance < closest:
                    closest = distance
            museums.append(closest)
        data["distance_closest_museum"] = museums
        return data
    
    def distance_to_closest_stadium(self,data):
        with open("stadiums.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            stadiums = []
        for _, row in data.iterrows():
            closest = float("inf")
            for stadium in coords:
                distance = self.distance(row["latitude"], row["longitude"], stadium[0], stadium[1])
                if distance < closest:
                    closest = distance
            stadiums.append(closest)
        data["distance_closest_stadium"] = stadiums
        return data
    
    def distance_to_closest_theater(self,data):
        with open("theaters.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            theaters = []
        for _, row in data.iterrows():
            closest = float("inf")
            for theater in coords:
                distance = self.distance(row["latitude"], row["longitude"], theater[0], theater[1])
                if distance < closest:
                    closest = distance
            theaters.append(closest)
        data["distance_closest_theater"] = theaters
        return data
    
    def distance_to_closest_church(self,data):
        with open("churches.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            churches = []
        for _, row in data.iterrows():
            closest = float("inf")
            for church in coords:
                distance = self.distance(row["latitude"], row["longitude"], church[0], church[1])
                if distance < closest:
                    closest = distance
            churches.append(closest)
        data["distance_closest_church"] = churches
        return data
    
    def distance_to_closest_railway(self,data):
        with open("railway_terminals.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            railways = []
        for _, row in data.iterrows():
            closest = float("inf")
            for railway in coords:
                distance = self.distance(row["latitude"], row["longitude"], railway[0], railway[1])
                if distance < closest:
                    closest = distance
            railways.append(closest)
        data["distance_closest_railway"] = railways
        return data
    
    def divide_logprice_by_total_area(self,data):
        data["price"] = data["price"]/data["area_total"]
        return data

    def distance_from_vnukovo(self,data):
        data["distance_from_vnukovo_airport"] = [self.distance(data["latitude"][i], data["longitude"][i],55.596111, 37.2675) for i in range(len(data["latitude"]))]
        return data
    
    def distance_from_sheremetyevo(self,data):
        data["distance_from_sheremetyevo_airport"] = [self.distance(data["latitude"][i], data["longitude"][i],55.972778, 37.414722) for i in range(len(data["latitude"]))]
        return data
    
    def distance_from_domodedovo(self,data):
        data["distance_from_domodedovo_airport"] = [self.distance(data["latitude"][i], data["longitude"][i],55.408611, 37.906111) for i in range(len(data["latitude"]))]
        return data
    
    def distance_from_zhukovsky(self,data):
        data["distance_from_zhukovsky_airport"] = [self.distance(data["latitude"][i], data["longitude"][i],55.553333, 38.151667) for i in range(len(data["latitude"]))]
        return data
    
    def distance_from_ostafyevo(self,data):
        data["distance_from_ostafyevo_airport"] = [self.distance(data["latitude"][i], data["longitude"][i],55.511667, 37.507222) for i in range(len(data["latitude"]))]
        return data

    def distance_to_closest_airport(self,data):
        with open("airports.txt", "r+") as file:
            lines = file.readlines()
            coords = [(float(line.split(",")[0]), float(line.split(",")[1])) for line in lines]
            airports = []
        for _, row in data.iterrows():
            closest = float("inf")
            for airport in coords:
                distance = self.distance(row["latitude"], row["longitude"], airport[0], airport[1])
                if distance < closest:
                    closest = distance
            airports.append(closest)
        data["distance_closest_airport"] = airports
        return data

    def multiply_square_price_with_total_area(self,data):
        data["price"] = data["price"]*data["area_total"]
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

    def find_rich_neighboors(self, data, test=False):
        path = 'neighboors_test.pickle' if test else 'neighboors_train.pickle'
        with open(path, 'rb') as file:
            rich_neighboors_loaded = load(file)

        data["rich_neighboors"] = rich_neighboors_loaded
        return data

    def combine_windows(self, data, boolean=False):
        has_windows = []
        for _, row in data.iterrows():
            if boolean:
                has_windows.append(row["windows_court"]
                                   >= 1 or row["windows_street"] >= 1)
                continue
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
        district_average = self.district_avg(data)
        self.district_avg_dict = district_average
        area_living = data[data["area_living"].notna()]
        area_living_fraction = (
            area_living["area_living"] / area_living["area_total"]).mean()
        area_kitchen = data[data["area_living"].notna()]
        area_kitchen_fraction = (
            area_kitchen["area_kitchen"] / area_kitchen["area_total"]).mean()
        
        data["district"] = data["district"].fillna(-1)
        for i, row in data.iterrows():
            key = str(
                int(row["district"] if not np.isnan(row["district"]) else -1))
            if np.isnan(row["latitude"]):
                data["latitude"][i] = district_average[key]["lat"]
            if np.isnan(row["longitude"]):
                data["longitude"][i] = district_average[key]["lon"]
            if np.isnan(row["area_kitchen"]):
                data["area_kitchen"][i] = area_kitchen_fraction * \
                    data["area_total"][i]
            if np.isnan(row["area_living"]):
                data["area_living"][i] = area_living_fraction * \
                    data["area_total"][i]
            if np.isnan(row["bathrooms_shared"]):
                data["bathrooms_shared"][i] = district_average[key]["bathrooms_shared"]
            if np.isnan(row["bathrooms_private"]):
                data["bathrooms_private"][i] = district_average[key]["bathrooms_private"]
            if np.isnan(row["phones"]):
                data["phones"][i] = district_average[key]["phones"]
            if np.isnan(row["constructed"]):
                data["constructed"][i] = district_average[key]["constructed"]
        data["seller"] = data["seller"].fillna(-1)
        data["windows_court"] = data["windows_court"].fillna(-1)
        data["windows_street"] = data["windows_street"].fillna(-1)
        data["new"] = data["new"].fillna(-1)
        data["material"] = data["material"].fillna(-1)
        data["elevator_passenger"] = data["elevator_passenger"].fillna(-1)
        data["elevator_service"] = data["elevator_service"].fillna(-1)
        data["parking"] = data["parking"].fillna(-1)
        data["garbage_chute"] = data["garbage_chute"].fillna(-1)
        data["heating"] = data["heating"].fillna(-1)
        return data

    def remove_zero_values(self, data, key):
        for i, row in data.iterrows():
            if row[key] == 0:
                data[key][i] = self.district_avg_dict[str(
                    int(row["district"]))][key]
        return data

    def general_removal(self, data):
        data = self.remove_labels(
            data, ["layout", "ceiling", "balconies", "loggias", "condition", "elevator_without", "street", "address"])
        return data

    def fix_latlon_outliers(self, data, outliers):
        for _, row in outliers.iterrows():
            data["latitude"][row["id"]] = self.district_avg_dict["4"]["lat"]
            data["longitude"][row["id"]] = self.district_avg_dict["4"]["lon"]
        return data

    def remove_redundant_features(self, data):
        return data

    def combine_latlon(self, data):
        data["distance_center"] = [self.distance(
            data["latitude"][i], data["longitude"][i]) for i in range(len(data["latitude"]))]
        return data
    
    def combine_latlon_subway(self, data, read_from_file=True):
        if(not read_from_file):
            data["closest_subway_distance"] = 0 
            subway_table = pd.read_csv("subway_table.csv")
            subway_distances = []
            for i in data.index:
                closest = float("inf")
                for j in (subway_table.index):
                    sub_way_distance = self.distance(data.at[i,"latitude"], data.at[i,"longitude"], subway_table.at[j,"latitude"], subway_table.at[j,"longitude"])
                    if(sub_way_distance < closest):
                        closest = sub_way_distance
                subway_distances.append(closest)
            data["closest_subway_distance"] = subway_distances
            save_to_file = pd.DataFrame(data["closest_subway_distance"])
            save_to_file.to_csv("./closest_subway_distance.csv", index=False)
        else:
            data["closest_subway_distance"]= pd.read_csv("./closest_subway_distance.csv")
        return data

    def combine_elevators(self, data):
        has_elevator = []
        for _, row in data.iterrows():
            has1 = row["elevator_passenger"] == 1
            has2 = row["elevator_service"] == 1
            if has1 and has2:
                has_elevator.append(2)
            elif has1 or has2:
                has_elevator.append(1)
            else:
                has_elevator.append(0)
        data["has_elevator"] = has_elevator
        return data

    def redo_new(self, data):
        is_new = [1 if row["constructed"] >=
                  2018 else 0 for _, row in data.iterrows()]
        data["new"] = is_new
        return data

    def distance(self, lat, lon, lat_to=55.754093, lon_to=37.620407):
        radius = 6373.0  # Earth radius in km
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


def main():
    p = Preprocessor()
    p.preprocess(p.merged)


#main()
