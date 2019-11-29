import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from scipy import stats



def generate_artificial_data(x, y, n_clust, proportion,seed):

    target = 'alcopops'

    df_x = x.copy()
    df_y = y.copy()

    df_concat = pd.concat([df_x, df_y], axis=1)
    # Generate clusters ************************************************************************************************

    df_to_clust = df_concat.copy()
    cluster_range = range(1, n_clust + 1)
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

    random.seed(seed)
    columns = centroids_df.columns.values
    columns = np.append(columns, 'cluster')
    artificial_points = pd.DataFrame(columns=columns)

    for j in range(len(centroids_df)):

        rows = int(len(df_to_clust.loc[df_to_clust['cluster_labels'] == j]) * proportion)

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


    artificial_concat = pd.concat([df_to_clust, artificial_points], axis = 0)
    x_return = artificial_concat.drop(columns = target)
    y_return = artificial_concat[target]

    return x_return,y_return