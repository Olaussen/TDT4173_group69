import pandas as pd # library for data analysis
import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents

class Mordi:
    def __init__(self,read = False):
        if(read):
            # get the response in the form of html
            wikiurl="https://en.wikipedia.org/wiki/List_of_Moscow_Metro_stations"
            table_class="wikitable sortable jquery-tablesorter"
            response=requests.get(wikiurl)
            print(response.status_code)
            # parse data from the html into a beautifulsoup object
            soup = BeautifulSoup(response.text, 'html.parser')
            tables= soup.findAll('table',{'class':"wikitable"})
            table = soup.find('table', class_='wikitable sortable')
            df=pd.read_html(str(table))
            df=pd.DataFrame(df[0])
            rows = df["Coordinates"]
            coords = [row.split("/")[1].strip() for row in rows]
            subway_coords = [(float(coord.split(" ")[0][1:8]),float(coord.split(" ")[1][:7])) for coord in coords]
            self.subway_table = pd.DataFrame(subway_coords, columns =['latitude', 'longitude'])
            self.subway_table.to_csv("./subway_table.csv",index=False)
        else:
            self.subway_table = pd.read_csv("./subway_table.csv")
            
        