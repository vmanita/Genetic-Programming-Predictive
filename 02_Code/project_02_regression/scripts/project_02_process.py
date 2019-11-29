import pandas as pd
import numpy as np
import scipy.stats as stats
import utils



def process_data(df_train, df_test, deal_nulls = 0, deal_outliers = 0):

    target = 'alcopops'
    train = df_train.copy()
    test = df_test.copy()

    # impute missings **************************************************************************************************
    # 0: Drop missings
    # 1: Mean or Median depending on distribution

    if deal_nulls == 0:
        train.dropna(inplace = True)
        test.replace(np.NaN, train.mean(), inplace = True)

    elif deal_nulls == 1:
        # see data distribution
        for column in train.drop(columns = target):
            shapiro_t, pvalue = stats.shapiro(train[column])

            if pvalue > 0.05:
                measure = train[column].median()
            else:
                measure = train[column].mean()

            train[column].replace(np.NaN, measure, inplace = True)
            test[column].replace(np.NaN, measure, inplace=True)

    # impute outliers **************************************************************************************************

    exclude = ['gender', 'year_collect', target]
    column_names = train.columns
    columns_to_search = [column for column in column_names if column not in exclude]

    if deal_outliers == 0:
        pass
    elif deal_outliers == 1:
        # winsoring
        p1 = 5
        p2 = 95
        train = utils.winsoring_smooth(train, columns_to_search, p1, p2)
    return train, test
