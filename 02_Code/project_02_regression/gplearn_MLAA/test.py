def main():
    import os
    import logging
    import datetime
    import numpy as np
    #from sklearn.datasets import load_boston
    from sklearn.metrics import mean_absolute_error
    from gplearn_MLAA.genetic import SymbolicRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    seed = 0

    path = 'C:/Users/rodri/Desktop/Nova IMS/Mestrado/1º Ano/2º Semestre/Machine Learning/Project 2/alcohol_children.xlsx'

    import pandas as pd

    data = pd.read_excel(path, index_col=0)

    # drop na
    # remove country and code

    data.dropna(inplace=True)

    data.drop(columns=['country', 'code'], inplace=True)

    train, test = train_test_split(data[:100], test_size=0.1, random_state=0)

    target = 'alcopops'

    x_train = train.drop(columns=target)
    y_train = train[target]

    x_test = test.drop(columns=target)
    y_test = test[target]

    # Scale **************

    scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_train_scaled = pd.DataFrame(x_train_scaled, columns = x_train.columns, index = x_train.index)

    x_test_scaled = scaler.transform(x_test)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns = x_test.columns, index = x_test.index)

    x_train = x_train_scaled.copy()
    x_test = x_test_scaled.copy()


    print("The baseline: {:.3f}".format(mean_absolute_error(y_test, np.repeat(y_test.mean(), len(y_test)))))

    # setup logger
    name = "project_draft"

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "logFiles/" + str(datetime.datetime.now().date()) + name + "log.txt")
    logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')

    edda_params = {"deme_size": 50, "p_gsgp_demes": 0.0, "maturation": 5, "p_mutation": 1.0, "gsm_ms": -1.0}

    est_gp = SymbolicRegressor(population_size=10, init_method='half and half',
                               generations=20, edda_params=None, stopping_criteria=0.0,
                               edv_stopping_criteria=0.0, n_semantic_neighbors=0, p_swap_mutation=0.0,
                               p_crossover=0.0,p_p2_crossover=0.0, p_uniform_crossover=0.7, p_subtree_mutation=0.0,
                               p_shake_mutation=0.3, p_graft_mutation=0.0, p_gs_crossover=0.0, p_gs_mutation=0.0,
                               gsm_ms=-1, semantical_computation=False, p_reverse_mutation=0.0,
                               p_hoist_mutation=0.0, p_simple_crossover=0.0, p_point_mutation=0.0,
                               parsimony_coefficient=0.001, function_set=('add', 'sub', 'mul', 'div',
                                                                        'sqrt', 'log', 'max', 'min',
                                                                        'power', 'log', 'exp', 'abs', 'neg'),
                               val_set=0.2,
                               verbose=1, n_jobs=1, log=True, random_state=seed)

    # print GS-GP
    est_gp.fit(x_train, y_train)
    if not est_gp.semantical_computation:
        print(est_gp._program)
        print("Generalization ability: {:.3f}".format(mean_absolute_error(y_test, est_gp.predict(x_test))))

if __name__ == "__main__":
    main()

"""_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 "tanh": tanh1,
                 'power': power1,
                 'exp': exp1}"""
