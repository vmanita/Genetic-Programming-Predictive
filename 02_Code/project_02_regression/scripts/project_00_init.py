from project_01_loader import import_dataset
from project_02_process import process_data
from project_03_features import feature_engineer
from project_03_features_artificial import generate_artificial_data
from project_04_grid_search import find_best_params
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import BaggingRegressor
import pandas as pd
import logging
import datetime
import warnings
from gplearn_MLAA.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
import numpy as np
import os
import utils
pd.set_option('display.max_columns', 6)

def main():
    # Load data ********************************************************************************************************

    file_path = "/Users/Manita/OneDrive - NOVAIMS/machine learning/Project_02/data/alcohol_children.xlsx"
    path_to_save = "/Users/Manita/OneDrive - NOVAIMS/machine learning/Project_02/backup/"
    data, unseen = import_dataset(file_path)

    # Split train test *************************************************************************************************

    test_size = 0.1

    n_runs = 20
    global_seeds = range(0, n_runs)

    # model with generalization ability over all seeds

    models_dict = {'RandomForestRegressor': [],
                   'GradientBoostingRegressor': [],
                   'AdaBoostRegressor': [],
                   'BaggingRegressor': []}


    gp_error = []
    for global_seed in global_seeds:

        train, test = train_test_split(data, test_size = test_size, random_state = global_seed)

        print('\n>>> CURRENT GLOBAL SEED:', global_seed)
        print("\nTrain df size:\t{} Observations\nTest df size:\t{} Observations".format(train.shape[0], test.shape[0]))
        # process the data
        deal_nulls = 1
        deal_outliers = 1
        train, test = process_data(train, test, deal_nulls = deal_nulls, deal_outliers = deal_outliers)

        # feature engineering
        # ******************
        n_features = 5
        variable_selection = False
        decomposition = False
        # ******************
        x_train, y_train, x_test, y_test = feature_engineer(train, test, global_seed, n_features,
                                                            variable_selection = variable_selection,
                                                            decomposition = decomposition)
        # Generate artificial data
        # ******************
        artificial = False
        n_clust = 15
        proportion = 0.8
        # ******************

        if artificial:
            print('n observations before artificial: {}'.format(x_train.shape[0]))
            x_train, y_train = generate_artificial_data(x_train, y_train, n_clust =n_clust, proportion=proportion, seed=global_seed)
            print('n observations after artificial: {}'.format(x_train.shape[0]))

        # grid search models *******************************************************************************************

        CV = 5

        rf_tuned = find_best_params(RandomForestRegressor(), x_train, y_train, global_seed, CV)
        gb_tuned = find_best_params(GradientBoostingRegressor(), x_train, y_train, global_seed, CV)
        ada_tuned = find_best_params(AdaBoostRegressor(), x_train, y_train, global_seed, CV)
        bag_tuned = find_best_params(BaggingRegressor(), x_train, y_train, global_seed, CV)

        models = [rf_tuned, gb_tuned, ada_tuned, bag_tuned]


        for model in models:

            name = model.__class__.__name__
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            generalization_ability = mean_absolute_error(y_test, y_pred)
            print(name,'Generalization ability: {:.3f}'.format(generalization_ability))
            models_dict[name].append(generalization_ability)

        # GP ***********************************************************************************************************

        # setup logger
        # initial benchmark

        p_cross = 0.75
        p_mut = 1-p_cross

        #name = "initial_bench_cross_{}_mut_{}".format(cross, mut)
        #name = "initial_bench_cross_{}_{}_mut_{}_{}".format(cross, p_cross, mut, p_mut)
        testing = 'final_gp_2'
        name = "xx_{}".format(testing)
        file_path = path_to_save + "logFiles/" + name + "_log.txt"
        #file_path = path_to_save + "logFiles/" + str(datetime.datetime.now().date()) + name + "_log.txt"

        logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')


        edda_params = {"deme_size": 50, "p_gsgp_demes": 0.1, "maturation": 9, "p_mutation": 1.0, "gsm_ms": -1.0}



        est_gp = SymbolicRegressor(population_size=500,# init_method='half and half',
                                   generations=250, edda_params=edda_params, stopping_criteria=0.0,
                                   edv_stopping_criteria=0.0, n_semantic_neighbors=0, p_swap_mutation=0.0,
                                   p_crossover=p_cross, p_p2_crossover=0.0, p_uniform_crossover=0.0, p_subtree_mutation=0.0,
                                   p_shake_mutation=0.00, p_graft_mutation=0.0, p_gs_crossover=0.0, p_gs_mutation=0.0,
                                   gsm_ms=-1, semantical_computation=False, p_reverse_mutation=0.0,
                                   p_hoist_mutation=p_mut, p_simple_crossover=0.0, p_point_mutation=0.0,
                                   parsimony_coefficient=0.001, function_set=('add', 'sub', 'mul', 'div',
                                                                              'sqrt', 'log', 'max', 'min',
                                                                              'power', 'log', 'exp', 'abs', 'neg'),
                                   val_set=0.1,
                                   verbose=1, n_jobs=1, log=False, random_state=global_seed)
        # print GS-GP
        est_gp.fit(x_train, y_train)
        if not est_gp.semantical_computation:
            print(est_gp._program)
            prediction = mean_absolute_error(y_test, est_gp.predict(x_test))
            print("Generalization ability: {:.3f}".format(prediction))
            gp_error.append(prediction)
    
    gp_error_df = pd.DataFrame({'seed':list(global_seeds),
                                'nmae':[ -x for x in gp_error]})

    gp_error_df.to_excel(path_to_save + "gp_box/" + name + ".xlsx")

    # export results to excel
    models_df = pd.DataFrame.from_dict(models_dict)
    excel_name = 'final_essembles'#.format(deal_nulls, deal_outliers, variable_selection, decomposition, artificial, n_clust, proportion)
    models_df.to_excel(path_to_save + '{}.xlsx'.format(excel_name))


if __name__ == "__main__":
    main()
