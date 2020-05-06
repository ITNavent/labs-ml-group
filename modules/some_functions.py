import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def sample_indices(df, test_pct = 0.15, replace = False, seed = 1):
    if seed != None:
        np.random.seed(seed)
    return np.random.choice(range(0, len(df)), round(test_pct*len(df)), replace=False)

def plot_cat_dist(df, cat, figsize = (15,3), n=25):
    category = df.groupby([cat], as_index=False).size().reset_index(name='count')
    category = category.sort_values(by='count',ascending=False).reset_index(drop=True)
    plt.figure(figsize = figsize)
    ax = sns.barplot(x=category.head(n)[cat], y=category.head(n)['count'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.title(cat)
    
def plot_cont_dist(df, var, figsize = (15,3), bins=25, low = None, high = None):
    if low==None : low=min(df[var])
    if high==None : high=max(df[var])
    plt.figure(figsize = figsize)
    sns.distplot(df[(df[var].notna())&
                 (df[var]>low)&(df[var]<high)][var]
                 , kde=False, rug=False, bins = bins)
    plt.ticklabel_format(style='plain', axis='x')
    plt.ticklabel_format(style='plain', axis='y')
    plt.title(var)