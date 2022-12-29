import pandas as pd
import numpy as np

from tqdm import tqdm

pd.set_option("display.max_columns", None)

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

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
    
def sample_latent_space(model, 
                        random=False, base_point=None, 
                        coor1=0, coor1_min=0, coor1_max=1, coor1_n=10, 
                        coor2=1, coor2_min=0, coor2_max=0, coor2_n=10,
                        outdim1=28, outdim2=28):
    
    disparray = np.zeros((outdim1*coor1_n,outdim2*coor2_n))                    
    coor1_trajectory = np.arange(coor1_min, coor1_max, (coor1_max - coor1_min)/coor1_n)
    coor2_trajectory = np.arange(coor2_min, coor2_max, (coor2_max - coor2_min)/coor2_n)
    sampled_point = base_point.copy()                    
    for k in range(coor1_n): 
        for j in range(coor2_n):
            sampled_point[0, coor1] = coor1_trajectory[k]
            sampled_point[0, coor2] = coor2_trajectory[k]

            preds = np.squeeze(model.predict(sampled_point, verbose=False))

            disparray[outdim1*k:outdim1*(k+1), outdim2*j:outdim2*(j+1)] = preds

    fig, axs = plt.subplots(1,1)
    axs.imshow(disparray, cmap='gray_r')
    plt.show()
