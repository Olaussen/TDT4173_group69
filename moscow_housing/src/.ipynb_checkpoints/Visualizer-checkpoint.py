from Preprocessor import Preprocessor 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:

    def __init__(self):
        self.preprocessor = Preprocessor()

    def area_to_price(self):
        data = self.preprocessor.apartments
        x = data["area_total"]
        y =  data["price"]
        plt.plot(x ,y, "o")
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x+b)
        plt.title("Total area to price")
        plt.xlabel("Area")
        plt.ylabel("Price")
        plt.show()
        return m, b
    
    def floor_to_price(self):
        data = self.preprocessor.apartments
        plt.scatter(data["price"], data["floor"])
        plt.title("Floor to price")
        plt.show()
