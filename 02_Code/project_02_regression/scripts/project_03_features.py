import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.patches as mpatches
import warnings
import matplotlib.ticker as mtick
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
import utils
import pandas.core.algorithms as algos
from pandas import Series
import re
import traceback
import string
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

def feature_engineer(df_train, df_test,seed, n_features=10, variable_selection = False, decomposition = False):

    train = df_train.copy()
    test = df_test.copy()

    target = 'alcopops'

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(train.drop(columns = target))
    x_train = pd.DataFrame(x_train, columns = train.drop(columns = target).columns, index = train.index)
    y_train = train[target]

    x_test = scaler.transform(test.drop(columns = target))
    x_test = pd.DataFrame(x_test, columns=test.drop(columns=target).columns, index=test.index)
    y_test = test[target]

    # Variable Selection ***********************************************************************************************
    if variable_selection:
        # Recursive Feature Elimination
        model = LinearRegression()
        rfe = RFE(model, n_features)
        fit = rfe.fit(x_train, y_train)
        columns_to_keep = x_train.columns[fit.support_]
        x_train = x_train[columns_to_keep]
        x_test = x_test[columns_to_keep]
    else:
        pass


    # PCA **************************************************************************************************************
    if decomposition:
        n_components = x_train.shape[1]
        i = 1
        achieved = False
        for n in range(n_components):
            pca = PCA(n_components=i)
            principalComponents = pca.fit_transform(x_train)
            cumulative_var = np.sum(pca.explained_variance_ratio_ * 100)
            if cumulative_var >= 80 and achieved == False:
                threshold_80_percent = i
                achieved = True
                print('\n>>> PCA 80% of cumulative explained Variance achieved with: {} components\n'.format(i))
                break
            i += 1

        pca_index = []
        for x in range(1, i + 1):
            pca_index.append('PC' + str(x))

        pca_df = pd.DataFrame(principalComponents, columns=pca_index, index=x_train.index).iloc[:, : threshold_80_percent]
        pca_test = pca.transform(x_test)
        pca_test_df = pd.DataFrame(pca_test, columns=pca_index, index=x_test.index).iloc[:, : threshold_80_percent]

        x_train = pca_df.copy()
        x_test = pca_test_df.copy()

    return x_train, y_train, x_test, y_test











