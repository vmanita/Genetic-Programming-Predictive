import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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
import pandas.core.algorithms as algos
from pandas import Series
import re
import traceback
import string
import random
from sklearn.cluster import KMeans

#**************************************************************************************************************
# Visualization
#**************************************************************************************************************

def missing_values_reporter(df):
    na_count = df.isna().sum()
    ser = na_count[na_count > 0]
    ser_p = np.round(ser.divide(df.shape[0])*100,2)
    tmp = pd.DataFrame({"N missings": ser,"% missings": ser_p,"Above Threshold (3%)": False})
    tmp.loc[tmp["% missings"] > 3., 'Above Threshold (3%)'] = 'True'
    return tmp


def plot_missing(df_miss, cutoff_ = 3):
    cutoff_list = [cutoff_, cutoff_]
    #plt.figure(figsize=(15,5))
    ax = df_miss.sort_values('% missings', ascending=False).plot.bar(y="% missings",
                                                                     color="Grey",
                                                                     alpha = 0.9,
                                                                     title="{}% cutoff line on missing values".format(cutoff_),
                                                                     legend=False,figsize = (10,4))
    ax.set_xlabel("Features with missing values", size=12)
    ax.set_ylabel("Proportion of missings")
    ax.plot([-1, len(df_miss.index)], cutoff_list,'r--', lw=2)
    ax.set_xticklabels(df_miss.sort_values('% missings', ascending=False).index, rotation=60, size=9)
    plt.show()

def plot_importance(df,measure, top = 20):
    to_plot = pd.DataFrame(df[measure]).head(top).sort_values(by=measure)

    # 1 font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'

    # 2 axis style

    plt.rcParams['axes.edgecolor']='white'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='white'
    plt.rcParams['ytick.color']='white'

    # plot
    my_range=range(1,len(to_plot.index)+1)
    fig, ax = plt.subplots(figsize=(10,10))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.hlines(y=my_range, xmin=0, xmax = to_plot[measure], color='white', alpha=0.4)
    plt.plot(to_plot[measure], my_range, "o", markersize=6, color='white', alpha=0.6)
    plt.yticks(my_range, to_plot.index,fontsize=10, fontweight = 'bold')
    # set labels style
    ax.set_title(measure, fontweight = 'bold')
    ax.set_xlabel('Importance', fontsize=10, fontweight='black', color = 'white')
    ax.set_ylabel('')
    # change the style of the axis spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)




# **************************************************************************************************************
# Var importance
# **************************************************************************************************************

def plot_importance(df,measure, top = 20):
    to_plot = pd.DataFrame(df[measure]).head(top).sort_values(by=measure)

    # 1 font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'

    # 2 axis style

    plt.rcParams['axes.edgecolor']='black'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='black'
    plt.rcParams['ytick.color']='black'

    # plot
    my_range=range(1,len(to_plot.index)+1)
    fig, ax = plt.subplots(figsize=(10,10))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.hlines(y=my_range, xmin=0, xmax = to_plot[measure], color='black', alpha=0.4)
    plt.plot(to_plot[measure], my_range, "o", markersize=6, color='black', alpha=0.6)
    plt.yticks(my_range, to_plot.index,fontsize=10, fontweight = 'bold')
    # set labels style
    ax.set_title(measure, fontweight = 'bold')
    ax.set_xlabel('Importance', fontsize=10, fontweight='black', color = 'black')
    ax.set_ylabel('')
    # change the style of the axis spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)


def chisq_ranker(df, continuous_flist, target, categorical_flist=None, n_bins=10, binning_strategy="uniform", ):
    chisq_dict = {}
    if continuous_flist:
        bindisc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                                   strategy=binning_strategy)
        for feature in continuous_flist:
            feature_bin = bindisc.fit_transform(df[feature].values[:, np.newaxis])
            feature_bin = pd.Series(feature_bin[:, 0], index=df.index)
            cont_tab = pd.crosstab(feature_bin, df[target], margins=False)
            chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2]
    if categorical_flist:
        for feature in categorical_flist:
            cont_tab = pd.crosstab(df[feature], df[target], margins=False)
            chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2]

    return chisq_dict


max_bin = 10
force_bin = 3


# define a binning function
def mono_bin(Y, X, n=max_bin):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1] - (bins[1] / 2)
        d1 = pd.DataFrame(
            {"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True)})
        d2 = d1.groupby('Bucket', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()

    return (d3)


def char_bin(Y, X):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    df2 = notmiss.groupby('X', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)

    return (d3)


def data_vars(df1, target):
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]

    x = df1.dtypes.index
    count = -1

    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1

            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv, ignore_index=True)

    iv = pd.DataFrame({'IV': iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return (iv_df, iv)


def winsoring_smooth(df, columns_to_search, p1, p2):
    smooth_df = df.copy()
    for feature in columns_to_search:
        q1 = np.percentile(smooth_df[feature], p1)
        q2 = np.percentile(smooth_df[feature], p2)

        # smooth
        smooth_df.loc[smooth_df[feature] > q2, feature] = q2
        smooth_df.loc[smooth_df[feature] < q1, feature] = q1
    return smooth_df




def generate_artificial(centroids_df, df, target,percent, seed):
    df_to_clust = df.copy()
    random.seed(seed)
    columns = centroids_df.columns.values
    columns = np.append(columns, 'cluster')
    artificial_points = pd.DataFrame(columns=columns)

    for j in range(len(centroids_df)):

        rows = int(len(df_to_clust.loc[df_to_clust['cluster_labels'] == j]) * percent)

        for x in range(rows):

            list1 = centroids_df.iloc[j].values
            list2 = df_to_clust.loc[df_to_clust['cluster_labels'] == j].iloc[x]
            point = []
            for i in range(len(list1)):
                point.append(random.uniform(list1[i], list2[i]))
            point.append(j)
            artificial_points.loc[len(artificial_points)] = point

        artificial_points['cluster'] = artificial_points['cluster'].astype(int)

        variables = ['gender', 'year_collect']

        for variable in variables:
            counter = 0
            while counter < len(centroids_df):
                mode = stats.mode(df_to_clust.loc[df_to_clust['cluster_labels'] == counter, variable].values)[0]
                artificial_points.loc[artificial_points['cluster'] == counter, variable] = mode
                counter += 1

        artificial_points['gender'] = artificial_points['gender'].astype(int)

    df_to_clust.drop(columns='cluster_labels', inplace=True)
    artificial_points.drop(columns='cluster', inplace=True)

    reverse_scaling = pd.DataFrame(artificial_points,
                                   columns=artificial_points.columns,
                                   index=artificial_points.index)

    artificial_df = artificial_points.drop(columns=target)

    artificial_df = pd.concat([reverse_scaling[target], artificial_df], axis=1)

    return artificial_df


def get_artificial_df(n_clusters, df, percent, seed):
    df_to_clust = df.copy()
    cluster_range = range(1, n_clusters + 1)
    cluster_errors = []
    sse = {}
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k,
                        random_state=seed,
                        n_init=10,
                        max_iter=300).fit(df_to_clust)
        sse[k] = kmeans.inertia_
        cluster_errors.append(kmeans.inertia_)

    df_to_clust["cluster_labels"] = kmeans.labels_
    centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=df_to_clust.drop(columns=['cluster_labels']).columns)

    artificial_df = generate_artificial(centroids_df, df_to_clust, percent, seed)

    return artificial_df

def artificial_concat(df, artificial_df):
    return pd.concat([df,artificial_df], axis = 0)