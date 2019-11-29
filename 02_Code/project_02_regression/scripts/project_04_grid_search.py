from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def find_best_params(model, x_train, y_train,seed ,CV = 5):

    grid_dict_params = params_collection(seed)

    name = model.__class__.__name__

    #grid_search_dict = {}

    model_tuned = GridSearchCV(model, grid_dict_params[name], cv=CV, n_jobs=-1, scoring='neg_mean_absolute_error',
                            verbose=1)
    model_tuned.fit(x_train, y_train)

    return model_tuned.best_estimator_


def params_collection(seed):
    grid_dict = {

        'RandomForestRegressor': {'random_state': [seed],
                                  'criterion': ['mse'],
                                  'n_estimators': [10, 30, 50],
                                  'min_samples_leaf': [1, 2, 4],
                                  'min_samples_split': [2, 5],
                                  'max_depth': [5, 10, 20, 30],
                                  },

        'GradientBoostingRegressor': {'random_state': [seed],
                                      'learning_rate': [0.05, 0.1, 0.2],
                                      'n_estimators': [10, 30, 50],
                                      "min_samples_split": np.linspace(0.1, 0.5, 5),
                                       "min_samples_leaf": np.linspace(0.1, 0.5, 5),
                                       "max_depth": [3, 5],
                                       "max_features": ["log2", "sqrt"]
                                      },
        'AdaBoostRegressor': {'random_state': [seed],
                              'n_estimators': [10, 30, 50],
                              'learning_rate': [0.05, 0.1, 0.2],
                              'loss': ['linear','square','exponential']
                              },
        'BaggingRegressor': {'random_state': [seed],
                             'n_estimators': [10, 30, 50],
                             'bootstrap': [True, False],
                             'warm_start': [True, False]}#,
                             #'max_features': ["log2", "sqrt"]}

    }

    return(grid_dict)


