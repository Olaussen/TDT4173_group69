import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    
    def list_missing(self, data):
        cols = [col for col in data.columns]
        tot_rows = len(data)
        for col in cols:
            missing = data[col].isna().sum()
            print(f'{col}: {round(missing*100/tot_rows, 2)}%')

    def price_correlation(self, data, figsize=(20,15)):
        corr = data.corr()
        corr['abs'] = abs(corr["price"])
        cor_target = corr.sort_values(by='abs', ascending=False)["price"]
        del corr['abs']
        corr = corr.loc[cor_target.index, cor_target.index]
        plt.figure(figsize=figsize)
        ax = sns.heatmap(corr, cmap='RdBu_r', annot=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
    def plot_distribution(self, data, column, bins=50, correlation=None):
        plt.figure(figsize=(12,8))
        data[column].hist(bins=bins)
        if not correlation is None:
            value = correlation[column]
            column = column + f' - {round(value,2)}'
        plt.title(f'Distribution of {column}', fontsize=18)
        plt.grid(False)

    def plot_vs(self, data, x, y, hue=None, **kwargs):
        plt.figure(figsize=(12,8))
        sns.scatterplot(data=data, x=x, y=y, hue=hue, **kwargs)
        if hue:
            plt.title(f'{x} vs {y}, by {hue}', fontsize=18)
        else:
            plt.title(f'{x} vs {y}', fontsize=18)

