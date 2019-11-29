import numpy as np
import pandas as pd


def import_dataset(path):

    df = pd.read_excel(path, index_col=0)

    target = 'alcopops'

    # keep record of those with null on target to predict in the end
    unseen = df[df['alcopops'].isna()]

    # Delete missings in target variable
    df.dropna(axis=0, subset=[target], inplace=True)

    df.drop(columns = ['country','code'], inplace=True)
    unseen.drop(columns=['country', 'code'], inplace=True)

    return df, unseen

