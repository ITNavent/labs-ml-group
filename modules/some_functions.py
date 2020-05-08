import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def sample_indices(df, test_pct = 0.15, replace = False, seed = 1):
""" Toma una muestra de índices de la matriz 'df'
"""   
    if seed != None:
        np.random.seed(seed)
    return np.random.choice(range(0, len(df)), round(test_pct*len(df)), replace=replace)

def plot_cat_dist(df, cat, figsize = (15,3), n=25, title = None):
    """ Plotea la variable categorica 'cat' de las primeras 'n' líneas de la matriz 'df'
    """
    category = df.groupby([cat], as_index=False).size().reset_index(name='count')
    category = category.sort_values(by='count',ascending=False).reset_index(drop=True)
    plt.figure(figsize = figsize)
    ax = sns.barplot(x=category.head(n)[cat], y=category.head(n)['count'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    if title == None:
        title == cat
    plt.title(title)
    
def plot_cont_dist(df, var, figsize = (15,3), bins=25, low = None, high = None, title = None, kde = False, rug = False):
    """ Plotea la variable continua 'var' de la matriz 'df'
    """
    if low==None : low=min(df[var])
    if high==None : high=max(df[var])
    plt.figure(figsize = figsize)
    sns.distplot(df[(df[var].notna())&
                 (df[var]>low)&(df[var]<high)][var]
                 , kde=kde, rug=rug, bins = bins)
    plt.ticklabel_format(style='plain', axis='x')
    plt.ticklabel_format(style='plain', axis='y')
    if title == None:
        title == var
    plt.title(title)

def id_to_position(x, learner, cat):
    """ Funcion que toma una variable categórica y busca su posicion en la lista de classes del learner
    """
    try:
        classes = learner.data.dataset.x.classes[cat]
    except:
        print('No category ', cat)
        return None
    try:
        res = int(np.where(classes == str(x))[0][0])
    except:
        res = None
    return res

def position_to_id(position, learner, cat):
    """ Funcion que toma una posición en una matriz de embeddings y devuelve su nombre de clase
    """
    try:
        res = learner.data.dataset.x.classes[cat][position]
    except:
        res = None
    return res