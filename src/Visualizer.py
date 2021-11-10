import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    
    def list_missing(self, data):
        cols = [col for col in data.columns]
        tot_rows = len(data)
        print("Total length:", len(data))
        for col in cols:
            missing = data[col].isna().sum()
            print(f'{col}: Amount: {missing} | {round(missing*100/tot_rows, 2)}%')

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

    def plot_vs(self, data, x, y, reg=False,  figsize=(12, 8), hue=None,  **kwargs):
        plt.figure(figsize=figsize)
        sns.scatterplot(data=data, x=x, y=y, hue=hue, **kwargs)
        if hue:
            plt.title(f'{x} vs {y}, by {hue}', fontsize=18)
        else:
            plt.title(f'{x} vs {y}', fontsize=18)
        if reg:
            x = data[x]
            y = data[y]
            m, b = np.polyfit(x, y, 1)
            plt.plot(x, m*x + b)
            
    def corr_target(self, data, target, cols, x_estimator=None):
        print(data[cols+[target]].corr())
        num = len(cols)
        rows = int(num/2) + (num % 2 > 0)
        cols = list(cols)
        y = data[target]
        fig, ax = plt.subplots(rows, 2, figsize=(12, 5 * (rows)))
        i = 0
        j = 0
        for feat in cols:
            x = data[feat]
            if (rows > 1):
                sns.regplot(x=x, y=y, ax=ax[i][j], x_estimator=x_estimator)
                j = (j+1)%2
                i = i + 1 - j
            else:
                sns.regplot(x=x, y=y, ax=ax[i], x_estimator=x_estimator)
                i = i+1

    def plot_map(self, data, ax=None, s=5, a=0.75, q_lo=0.0, q_hi=0.9, cmap='autumn', column='price', title='Moscow apartment price by location'):
        data = data[['latitude', 'longitude', column]].sort_values(by=column, ascending=True)
        backdrop = plt.imread('../dataset/moscow.png')
        backdrop = np.einsum('hwc, c -> hw', backdrop, [0, 1, 0, 0]) ** 2
        if ax is None:
            plt.figure(figsize=(12, 8), dpi=100)
            ax = plt.gca()
        discrete = np.expm1(data[column]).nunique() <= 20
        if not discrete:
            lo, hi =  np.expm1(data[column]).quantile([q_lo, q_hi])
            hue_norm = plt.Normalize(lo, hi)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
            sm.set_array([])
        else:
            hue_norm = None 
        ax.imshow(backdrop, alpha=0.5, extent=[37, 38, 55.5, 56], aspect='auto', cmap='bone', norm=plt.Normalize(0.0, 2))
        sns.scatterplot(x='longitude', y='latitude', hue= np.expm1(data[column]).tolist(), ax=ax, s=s, alpha=a, palette=cmap,linewidth=0, hue_norm=hue_norm, data=data)
        ax.set_xlim(37, 38)    # min/max longitude of image 
        ax.set_ylim(55.5, 56)  # min/max latitude of image
        if not discrete:
            ax.legend().remove()
            ax.figure.colorbar(sm)
        ax.set_title(title)
        return ax, hue_norm

