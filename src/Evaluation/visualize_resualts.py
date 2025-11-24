#Vislauzie using plots for distribvutions and heatmap

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(series: pd.Series, title: str = None, bins: int = 30):
    plt.figure(figsize=(6,4))
    plt.hist(series.dropna(), bins=bins)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    if title:
        plt.title(title)
    plt.show()

def plot_boxplot(series: pd.Series, title: str = None):
    plt.figure(figsize=(6,4))
    plt.boxplot(series.dropna(), vert=False)
    plt.xlabel(series.name)
    if title:
        plt.title(title)
    plt.show()

#correlaation heatmap
def plot_heatmap(df: pd.DataFrame, method: str = "pearson"):
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(method=method)
    plt.figure(figsize=(10,8))
    plt.imshow(corr.values, interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(f"Correlation matrix ({method})")
    plt.tight_layout()
    plt.show()

def plot_value_count(series: pd.Series, title: str = None):
    vc = series.value_counts(dropna=False)
    plt.figure(figsize=(6, 4))
    plt.bar(vc.index.astype(str), vc.values)
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()