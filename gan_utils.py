#sources:
#https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/
#https://blog.keras.io/building-autoencoders-in-keras.html
#sources: https://sinyi-chou.github.io/classification-pr-curve/
#! pip install arrow

import pandas as pd
import numpy as np

from tqdm import tqdm

pd.set_option("display.max_columns", None)

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

import tensorflow as tf
from tensorflow.keras.optimizers import SGD

import sklearn
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import graphviz


import pickle


def scale_data(x):
    #scaling the input:
    scaler = StandardScaler()
    scaler.fit(x)
    display(scaler.mean_)
    display(scaler.scale_)
    scaleddata = scaler.transform(x)
    return scaler, scaleddata

def draw_distribution(real_df, fake_df, bins=10):
    for col in real_df.columns:
        comparison_df = pd.DataFrame({col + '_real': real_df[col], 
                                      col + '_fake': fake_df[col]})
        comparison_df.hist(bins=bins)
        
def pairwise_corr(real_df, fake_df):
    f, (axarr_real, axarr_fake) = plt.subplots(1, 2)
    f.suptitle('pairwise feature corrs, real (left), fake (right)')
    #cb = plt.colorbar()
    axarr_real.matshow(real_df.corr())
    axarr_fake.matshow(fake_df.corr())
