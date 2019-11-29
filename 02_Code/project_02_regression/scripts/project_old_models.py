import pandas as pd
import numpy as np
import project_00_init

#******************************************************************************
# Mandatory models
#******************************************************************************

df_processed = df.drop(columns = ['country','code'])


x = df_processed.drop(columns = 'alcopops')
y = df_processed['alcopops']


# Random Forest

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=2, random_state=seed,n_estimators=100)

model.fit(x, y)

predictions = model.predict(x)

# Adaptive Boosting

from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor(random_state=seed, n_estimators=100)

model.fit(x, y)

predictions = model.predict(x)


# Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(learning_rate = 0.1,random_state = seed)

model.fit(x, y)

predictions = model.predict(x)


#******************************************************************************
# Mandatory models
#******************************************************************************

from sklearn.metrics import mean_squared_error

mean_squared_error(y, predictions)


def var_importance(train, target, seed):

    # Split data into features and Labels
    x = train.drop(columns=target)
    y = train[target]

    # Normalize Data
    scaler = MinMaxScaler()

    x_ = scaler.fit_transform(x)
    x = pd.DataFrame(x_, columns=x.columns, index=x.index)

    #*******
    # Chi2
    #*******

    continuous_flist = list(train.select_dtypes(include=["number"]).drop(columns=target).columns)

    chisq_rank = utils.chisq_ranker(train, continuous_flist, target)
    chisq_rank
    df_chisq_rank = pd.DataFrame(chisq_rank, index=["Chi-Squared", "p-value"]).transpose()
    df_chisq_rank.sort_values("Chi-Squared", ascending=False, inplace=True)
    df_chisq_rank["valid"] = df_chisq_rank["p-value"] <= 0.05
    # chi
    chi_sq = pd.DataFrame(df_chisq_rank['Chi-Squared']).rank(ascending=False).astype('int64')
    chi_sq.head()

    # *******
    # LinReg
    # *******

    r_squared = []
    coef = []
    p_val = []

    for var in x.columns:
        x_ = x[var]
        mod = sm.OLS(y, sm.add_constant(x_)).fit()
        r_squared.append(np.round(mod.rsquared, decimals=3))
        coef.append(np.round(mod.params[1], decimals=3))
        p_val.append(mod.pvalues[1])

    # DataFrame
    LinReg = pd.DataFrame({"variable": x.columns, "R2": r_squared, "coef": coef, 'pvalue': p_val})
    LinReg['valid'] = LinReg['pvalue'] <= 0.05
    LinReg.sort_values(by="R2", ascending=False, inplace=True)
    LinReg.set_index("variable", inplace=True)

    LinReg_ = pd.DataFrame(LinReg.R2)
    LinReg = pd.DataFrame(LinReg.R2).rank(ascending=False).astype('int64')
    LinReg.rename(index=str, columns={"R2": "LinReg"}, inplace=True)

    # *******
    # RandFor
    # *******

    # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    clf = RandomForestRegressor(random_state=seed)
    clf.fit(x, y)

    preds = clf.predict(x)

    rf_ = pd.DataFrame(clf.feature_importances_, columns=["RF"], index=x.columns)
    rf = pd.DataFrame(clf.feature_importances_, columns=["RF"], index=x.columns).rank(ascending=False).astype('int64')

    # *******
    # RFE
    # *******

    # Recursive Feature Elimination
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    rfe = RFE(model, 1)
    fit = rfe.fit(x, y)
    rfe = pd.DataFrame(rfe.ranking_, columns=["RFE"], index=x.columns)
    rfe_ = rfe.copy().sort_values(by='RFE', ascending=False)

    # *******
    # Extratr
    # *******

    from sklearn.ensemble import ExtraTreesRegressor
    model = ExtraTreesRegressor(random_state=seed)
    model.fit(x, y)
    extc_ = pd.DataFrame(model.feature_importances_, columns=["Extratrees"], index=x.columns)
    extc = pd.DataFrame(model.feature_importances_, columns=["Extratrees"], index=x.columns).rank(
        ascending=False).astype('int64')


    # *******
    # DT
    # *******

    from sklearn import tree
    from sklearn.tree import DecisionTreeRegressor
    dtree = DecisionTreeRegressor(random_state=seed)
    dtree = dtree.fit(x, y)
    dtree_ = pd.DataFrame(dtree.feature_importances_, columns=["Dt"], index=x.columns)
    dtree = pd.DataFrame(dtree.feature_importances_, columns=["Dt"], index=x.columns).rank(ascending=False).astype(
        'int64')


    # *******
    # WOE
    # *******

    final_iv, IV = utils.data_vars(train[train.columns.difference([target])], train[target])
    IV.set_index('VAR_NAME', inplace=True)
    IV_rank = IV.rank(ascending=False).astype('int64')

    transform_vars_list = train.columns.difference([target])
    transform_prefix = 'WOE_'

    df_woe = train.copy()

    for var in transform_vars_list:
        small_df = final_iv[final_iv['VAR_NAME'] == var]
        transform_dict = dict(zip(small_df.MAX_VALUE, small_df.WOE))
        replace_cmd = ''
        replace_cmd1 = ''
        for i in sorted(transform_dict.items()):
            replace_cmd = replace_cmd + str(i[1]) + str(' if x <= ') + str(i[0]) + ' else '
            replace_cmd1 = replace_cmd1 + str(i[1]) + str(' if x == "') + str(i[0]) + '" else '
        replace_cmd = replace_cmd + '0'
        replace_cmd1 = replace_cmd1 + '0'
        if replace_cmd != '0':
            try:
                df_woe[transform_prefix + var] = train[var].apply(lambda x: eval(replace_cmd))
            except:
                df_woe[transform_prefix + var] = train[var].apply(lambda x: eval(replace_cmd1))

    woe_columns = [x for x in df_woe.columns if x.startswith('WOE_')]

    df_woe = df_woe[woe_columns]
    df_woe = pd.concat([df_woe, train[target]], axis=1)

    from functools import reduce
    dfs = [chi_sq, LinReg, rf, rfe, extc, dtree, IV_rank]
    compare_models = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)

    # *********************
    # Compare Models: Vote
    # *********************

    compare_models['Ranking'] = compare_models.sum(axis=1).rank(ascending=True).astype('int64')
    compare_models.sort_values(by='Ranking', inplace=True)
    compare_models['Valid'] = df_chisq_rank.valid
    print(compare_models.head())