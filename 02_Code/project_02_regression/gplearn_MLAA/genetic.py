"""Genetic Programming in Python, with a scikit-learn inspired API

The :mod:`gplearn_MLAA.genetic` module implements Genetic Programming. These
are supervised learning methods based on applying evolutionary operations on
computer programs.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import itertools
from abc import ABCMeta, abstractmethod
from time import time
from warnings import warn
import logging
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.random import sample_without_replacement
from ._program import _Program
from .fitness import _fitness_map, _Fitness
from .functions import _function_map, _Function, sig1 as sigmoid
from .utils import _partition_estimators
from .utils import check_random_state, NotFittedError

__all__ = ['SymbolicRegressor', 'SymbolicClassifier', 'SymbolicTransformer']

MAX_INT = np.iinfo(np.int32).max



def _initialize_edda(params, population_size, X, y, sample_weight,
                     train_indices, val_indices, verbose, logger, random_state, n_jobs):
    # The population of local elites
    despeciation_pool = []

    # Prepare output generation for EDDA
    run_details_ = {"deme": [],
                    "gp_deme": [],
                    'generation': [],
                    'average_length': [],
                    'average_fitness': [],
                    'best_length': [],
                    'best_fitness': [],
                    'best_val_fitness': [],
                    'best_oob_fitness': [],
                    'generation_time': []}

    def _verbose_reporter_edda(run_details=None):
        """A report of the progress of demes evolution process.

        Parameters
        ----------
        run_details : dict
            Information about demes evolution.

        """
        if run_details is None:
            print('              |{:^25}|{:^59}|'.format("Deme Average",
                                               'Best Individual'))
            print('-' * 14 + ' ' + '-' * 25 + ' ' + '-' * 59 + ' ' + '-' * 10)
            line_format = '{:>5} {:>3} {:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>16} {:>10}'
            print(line_format.format("Deme", "GP", 'Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', "VAL Fitness", 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (params["edda_params"]["maturation"] - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            val_fitness = 'N/A'
            line_format = '{:>5d} {:>3d} {:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>16} {:>10}'
            if val_indices is not None:
                val_fitness = run_details['best_val_fitness'][-1]
                line_format = '{:>5d} {:>3d} {:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>16} {:>10}'

            oob_fitness = 'N/A'
            if sample_weight is not None:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:>5d} {:>3d} {:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['deme'][-1],
                                     run_details['gp_deme'][-1],
                                     run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     val_fitness,
                                     oob_fitness,
                                     remaining_time))

    # Print header for EDDA initialization
    if verbose:
        _verbose_reporter_edda()

    # Define number of GP and GSGP demes
    n_gsgp_demes = int(params["edda_params"]["p_gsgp_demes"] * population_size)

    for deme in range(population_size):
        parents = None

        # Copy algorithm's original parameters
        params_ = dict(params)
        if deme < n_gsgp_demes:
            # GS-GP
            params_['method_probs'] = np.array([0.0, 0.0, 0.0, 0.0, 1.0 - params["edda_params"]["p_mutation"],
                                                1.0, params["edda_params"]["gsm_ms"]])
        else:
            # Standard-GP
            params_['method_probs'] = np.array([1.0 - params["edda_params"]["p_mutation"], 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

        best_program = None
        for gen in range(params["edda_params"]["maturation"]):
            start_time = time()

            # Parallel loop
            n_jobs_, n_programs, starts = _partition_estimators(params["edda_params"]["deme_size"], n_jobs)
            seeds = random_state.randint(MAX_INT, size=params["edda_params"]["deme_size"])

            population = Parallel(n_jobs=n_jobs_, verbose=int(verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          train_indices,
                                          val_indices,
                                          seeds[starts[i]:starts[i + 1]],
                                          params_)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            parents = list(itertools.chain.from_iterable(population))

            fitness = [program.raw_fitness_ for program in parents]
            length = [program.length_ for program in parents]

            parsimony_coefficient_ = None
            if deme >= n_gsgp_demes and params['parsimony_coefficient'] == 'auto':
                parsimony_coefficient_ = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in parents:
                program.fitness_ = program.fitness(parsimony_coefficient_)

            # Record run details
            if params["_metric"].greater_is_better:
                best_program = parents[np.argmax(fitness)]
            else:
                best_program = parents[np.argmin(fitness)]

            # Store outputs
            run_details_['deme'].append(deme)
            run_details_['gp_deme'].append(int(deme >= n_gsgp_demes))
            run_details_['generation'].append(gen)
            run_details_['average_length'].append(np.mean(length))
            run_details_['best_length'].append(best_program.length_)
            run_details_['average_fitness'].append(np.mean(fitness))
            run_details_['best_fitness'].append(best_program.raw_fitness_)
            val_fitness = np.nan
            if val_indices is not None:
                val_fitness = best_program.val_fitness_
            run_details_["best_val_fitness"].append(val_fitness)
            oob_fitness = np.nan
            if sample_weight is not None:
                oob_fitness = best_program.oob_fitness_
            run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            run_details_['generation_time'].append(generation_time)

            if verbose:
                _verbose_reporter_edda(run_details_)

            if logger is not None:
                log_event = [detailsList[-1] for detailsList in run_details_.values()]
                logger.info(','.join(list(map(str, log_event))))

        despeciation_pool.append(best_program)

    return despeciation_pool


def _get_semantic_stopping_criteria(n_semantic_neighbors, elite, X, y, sample_weight, train_indices, params, seeds):
    n_samples, n_features = X[train_indices].shape
    # Unpack parameters
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']

    # Define the counter for number of better semantic neighbors
    n_better_semantic_neighbors = 0
    # Define counter for number of better semantic neighbors with smaller EDV
    n_better_semantic_neighbors_edv = 0
    # Compute ED for the elite
    if elite.semantical_computation:
        elite._ed = np.std(a=np.abs(np.subtract(elite.program[train_indices], y[train_indices])), ddof=1)
    else:
        elite._ed = np.std(a=np.abs(np.subtract(elite.execute(X[train_indices]), y[train_indices])), ddof=1)

    for i in range(n_semantic_neighbors):
        # Mutate elite by means of GS-M
        random_state = check_random_state(seeds[i])

        if elite.semantical_computation:
            program = elite.gs_mutation_tanh_semantics(X, random_state.uniform(), random_state)
        else:
            program = elite.gs_mutation_tanh(random_state.uniform(), random_state)

        # Create an instance of _Program, the semantic neighbor
        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program,
                           semantical_computation=elite.semantical_computation)

        # Evaluate the semantic neighbor
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight[train_indices].copy()

        indices, not_indices = program.get_all_indices(n_samples, max_samples, random_state)
        curr_sample_weight[not_indices] = 0

        if elite.semantical_computation:
            program.raw_fitness_ = program.metric(y[train_indices], program.program[train_indices], curr_sample_weight)
        else:
            program.raw_fitness_ = program.raw_fitness(X[train_indices], y[train_indices], curr_sample_weight)

        # If the neighbour is better than the elite, compare their EDVs
        if (metric.greater_is_better and (program.raw_fitness_ > elite.raw_fitness_)) or \
             (not metric.greater_is_better and (program.raw_fitness_ < elite.raw_fitness_)):
            # Add 1 to better semantic neighbors count
            n_better_semantic_neighbors += 1

            # Compute ED for the semantic neighbor
            if elite.semantical_computation:
                program._ed = np.std(a=np.abs(np.subtract(program.program[train_indices], y[train_indices])), ddof=1)
            else:
                program._ed = np.std(a=np.abs(np.subtract(program.execute(X[train_indices]), y[train_indices])), ddof=1)

            if program._ed < elite._ed:
                # Add 1 to smaller EDV of better semantic neighbors count
                n_better_semantic_neighbors_edv += 1

    tie = n_better_semantic_neighbors/n_semantic_neighbors
    edv = n_better_semantic_neighbors_edv/n_better_semantic_neighbors

    return tie, edv


def _parallel_evolve(n_programs, parents, X, y, sample_weight, train_indices, val_indices, seeds, params):
    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X[train_indices].shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']
    semantical_computation = params["semantical_computation"]

    max_samples = int(max_samples * n_samples)

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    def _control_bloat_selection():
        contenders = range(0, len(parents))
        lenghts = [parents[p].length_ for p in contenders]
        best_sample = (np.array(lenghts)).argsort()[:tournament_size]
        fitness = [parents[p].fitness_ for p in best_sample]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    def _tournament_tree_length():
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].length_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    def _tournament_tree_depth():

        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].depth_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    def _random_selection():
        individuals = range(0, len(parents))
        random_index = np.random.randint(0, len(parents))
        return parents[random_index], random_index

    def _roulette_wheel():

        individuals = range(0, len(parents))

        if metric.greater_is_better:
            fitness = [parents[p].fitness_ for p in individuals]
        else:
            fitness = [1 / (parents[p].fitness_) for p in individuals]

        total_fitness = np.sum(fitness)

        fitness_proportions = np.divide(fitness, total_fitness)
        cumulative_fitness_proportions = np.cumsum(fitness_proportions)

        threshold = random_state.uniform()

        indexes = np.argwhere(cumulative_fitness_proportions >= threshold)

        individual_to_return = indexes[0][0]

        return parents[individual_to_return], individual_to_return

    def _rank():

        individuals = range(0, len(parents))

        if metric.greater_is_better:
            fitness = np.array([parents[p].fitness_ for p in individuals])
            order = fitness.argsort()
            ranks = order.argsort() + 1
        else:
            fitness = np.array([1 / (parents[p].fitness_) for p in individuals])
            order = fitness.argsort()
            ranks = order.argsort() + 1

        total_rank = np.sum(ranks)

        rank_proportions = np.divide(ranks, total_rank)
        cumulative_rank_proportions = np.cumsum(rank_proportions)

        threshold = random_state.uniform()

        indexes = np.argwhere(cumulative_rank_proportions >= threshold)

        individual_to_return = indexes[0][0]

        return parents[individual_to_return], individual_to_return

    def _sus():

        sus_size = 5

        individuals = range(0, len(parents))

        if metric.greater_is_better:
            fitness = np.array([parents[p].fitness_ for p in individuals])
        else:
            fitness = np.array([1 / (parents[p].fitness_) for p in individuals])

        total_fitness = np.sum(fitness)
        point_distance = total_fitness / sus_size
        start_point = random_state.uniform(0, point_distance)
        pointers = [start_point + x * point_distance for x in range(sus_size)]

        keep = []
        for p in pointers:
            i = 1
            while np.sum(fitness[:i]) < p:
                i += 1
            keep.append(individuals[i - 1])

        # select first

        individual_to_return = keep[0]

        return parents[individual_to_return], individual_to_return

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            ##################
            sel = _rank()
            ##################
            method = random_state.uniform()
            parent, parent_index = sel

            if method < method_probs[0]:
                # GP: simple crossover
                donor, donor_index = sel
                program, removed, remains = parent.simple_crossover(donor.program, random_state)
                genome = {'method': 'Simple Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}

            elif method < method_probs[1]:
                # GP: swap crossover
                donor, donor_index = sel
                program, removed, remains = parent.crossover(donor.program, random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}

            elif method < method_probs[2]:
                # GP: p2 crossover
                donor1, donor_index1 = sel
                donor2, donor_index2 = sel
                program, removed, remains = parent.p2_crossover(donor1.program, donor2.program, random_state)
                genome = {'method': 'P2 Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index1,
                          'donor_nodes': remains}

            elif method < method_probs[3]:
                # GP: uniform crossover
                donor, donor_index = sel
                program, removed = parent.uniform_crossover(donor.program, random_state)
                genome = {'method': 'Uniform Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': removed}

            elif method < method_probs[4]:
                # GP: subtree mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[5]:
                # GP: hoist mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[6]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}

            elif method < method_probs[7]:
                # shake_mutation
                program, mutated = parent.shake_mutation(random_state)
                genome = {'method': 'Shake Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}

            elif method < method_probs[8]:
                # graft_mutation
                program, mutated = parent.graft_mutation(random_state)
                genome = {'method': 'Graft Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}

            elif method < method_probs[9]:
                # reverse_mutation
                program, mutated = parent.reverse_mutation(random_state)
                genome = {'method': 'Reverse Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}

            elif method < method_probs[10]:
                # swap_mutation
                program, mutated = parent.swap_mutation(random_state)
                genome = {'method': 'Swap Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}

            elif method < method_probs[11]:
                # GS-crossover
                donor, donor_index = sel#_tournament()
                if semantical_computation:
                    program = parent.gs_crossover_semantics(X, donor, random_state)
                else:
                    program = parent.gs_crossover(donor.program, random_state)

                genome = {'method': 'GS-Crossover',
                          'parent_idx': parent_index,
                          'donor_idx': donor_index}
            elif method < method_probs[12]:
                # GS mutation
                if method_probs[13] == -1:
                    gsm_ms = method
                else:
                    gsm_ms = method_probs[13]
                if semantical_computation:
                    program = parent.gs_mutation_tanh_semantics(X, gsm_ms, random_state)
                else:
                    program = parent.gs_mutation_tanh(gsm_ms, random_state)

                genome = {'method': 'GS-Mutation',
                          'parent_idx': parent_index}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program,
                           semantical_computation=semantical_computation)

        program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight[train_indices].copy()
        oob_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples, max_samples, random_state)

        curr_sample_weight[not_indices] = 0
        oob_sample_weight[indices] = 0

        if semantical_computation:
            if program.parents is None:
                # During initialization, an individual has to be a list of program elements to compute its semantcs
                program.program_length = program.length_
                program.program_depth = program.depth_
                program.program = program.execute(X)
            program.raw_fitness_ = program.metric(y[train_indices], program.program[train_indices], curr_sample_weight)
            if val_indices is not None:
                # Calculate validation fitness
                program.val_fitness_ = program.metric(y[val_indices], program.program[val_indices], None)
            if max_samples < n_samples:
                # Calculate OOB fitness
                program.oob_fitness_ = program.metric(y[train_indices], program.program[train_indices], oob_sample_weight)
        else:
            # Every individual is a list of program elements
            program.raw_fitness_ = program.raw_fitness(X[train_indices], y[train_indices], curr_sample_weight)
            if val_indices is not None:
                # Calculate validation fitness
                program.val_fitness_ = program.raw_fitness(X[val_indices], y[val_indices], None)
            if max_samples < n_samples:
                # Calculate OOB fitness
                program.oob_fitness_ = program.raw_fitness(X[train_indices], y[train_indices], oob_sample_weight)

        programs.append(program)

    return programs


class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):

    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 population_size=1000,
                 hall_of_fame=None,
                 n_components=None,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 tie_stopping_criteria=0.0,
                 edv_stopping_criteria=0.0,
                 n_semantic_neighbors=0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 edda_params=None,
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer=None,
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_simple_crossover=0.9,
                 p_p2_crossover=0.9,
                 p_uniform_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_shake_mutation=0.01,
                 p_graft_mutation=0.01,
                 p_reverse_mutation=0.01,
                 p_swap_mutation=0.01,
                 p_point_replace=0.05,
                 p_gs_crossover=0.0,
                 p_gs_mutation=0.0,
                 gsm_ms=-1.0,
                 semantical_computation=False,
                 val_set=0.0,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 log=False,
                 random_state=None):

        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.tie_stopping_criteria = tie_stopping_criteria
        self.edv_stopping_criteria = edv_stopping_criteria
        self.n_semantic_neighbors = n_semantic_neighbors
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.edda_params = edda_params
        self.function_set = function_set
        self.transformer = transformer
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_simple_crossover = p_simple_crossover
        self.p_p2_crossover = p_p2_crossover
        self.p_uniform_crossover = p_uniform_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_shake_mutation = p_shake_mutation
        self.p_graft_mutation = p_graft_mutation
        self.p_reverse_mutation = p_reverse_mutation
        self.p_swap_mutation = p_swap_mutation
        self.p_point_replace = p_point_replace
        self.p_gs_crossover = p_gs_crossover
        self.p_gs_mutation = p_gs_mutation
        self.gsm_ms = gsm_ms
        self.semantical_computation = semantical_computation
        self.val_set = val_set
        self.max_samples = max_samples
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.log = log
        self.random_state = random_state

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^59}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 59 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', "VAL Fitness", 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            val_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>16} {:>10}'
            if self.val_set > 0.0:
                val_fitness = run_details['best_val_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>16} {:>10}'

            oob_fitness = 'N/A'
            if self.max_samples < 1.0:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     val_fitness,
                                     oob_fitness,
                                     remaining_time))

    def fit(self, X, y, sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        # Check arrays
        if isinstance(self, ClassifierMixin):
            X, y = check_X_y(X, y, y_numeric=False)
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)
            n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
            if n_trim_classes != 2:
                raise ValueError("y contains %d class after sample_weight "
                                 "trimmed classes with zero weights, while 2 "
                                 "classes are required."
                                 % n_trim_classes)
            self.n_classes_ = len(self.classes_)
        else:
            X, y = check_X_y(X, y, y_numeric=True)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        _, self.n_features_ = X.shape

        # validation subset
        if self.val_set == 0.0:
            train_indices, val_indices = np.arange(X.shape[0]), None
        elif 0.0 < self.val_set < 1.0:
            val_indices = sample_without_replacement(X.shape[0], int(self.val_set*X.shape[0]),
                                                     random_state=self.random_state)
            sample_counts = np.bincount(val_indices, minlength=X.shape[0])
            train_indices = np.where(sample_counts == 0)[0]
        else:
            raise ValueError('val_set (:2.f) must be bigger than or equal to 0.0 and smaller than 1.0'.format(self.val_set))

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'log loss':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            if self.metric not in ('pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        # Parameters for standard GP operators
        self._method_probs = np.array([self.p_simple_crossover,
                                       self.p_crossover,
                                       self.p_p2_crossover,
                                       self.p_uniform_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation,
                                       self.p_shake_mutation,
                                       self.p_graft_mutation,
                                       self.p_reverse_mutation,
                                       self.p_swap_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] < 0.0 or self._method_probs[-1] > 1.0:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, p_hoist_mutation '
                             'and p_point_mutation must be in [0.0, 1.0]')

        # Parameters for GS operators
        _gs_method_probs = self.p_gs_crossover + self.p_gs_mutation

        if _gs_method_probs > 0.0 and self._method_probs[-1] > 0.0:
            raise ValueError('The user must to choose between standard GP and GS-GP operators.')

        if 0 < _gs_method_probs and _gs_method_probs != 1.0:
            raise ValueError('The sum of p_gs_crossover and p_gs_mutation must be equal to 1.0')

        self._method_probs = np.append(self._method_probs, np.array([self.p_gs_crossover, _gs_method_probs, self.gsm_ms]))

        # Parameters for semantic stopping criteria
        if 0.0 < self.tie_stopping_criteria <= 1 and 0 < self.edv_stopping_criteria <= 1.0:
            raise ValueError('Only one semantic stopping criteria is allowed: TIE or EDV. '
                             'Got TIE={0:.2f} EDV={1:.2f}.'.format(self.tie_stopping_criteria,
                                                                    self.edv_stopping_criteria))

        if 0.0 > self.tie_stopping_criteria or self.tie_stopping_criteria > 1.0:
            raise ValueError('TIE semantic stopping criteria should be between 0 and 1. Given {:.2f}'.format(
                self.tie_stopping_criteria))

        if 0.0 > self.edv_stopping_criteria or self.edv_stopping_criteria > 1.0:
            raise ValueError('EDV semantic stopping criteria should be between 0 and 1. Given {:.2f}'.format(
                self.EDV_stopping_criteria))

        if (0.0 < self.tie_stopping_criteria <= 1 or 0 < self.edv_stopping_criteria <= 1.0) and \
                self.n_semantic_neighbors<=0:
            raise ValueError('The number of semantic neighbors must be an integer value higher than 0. '
                             'Got n_semantic_neighbors={:}'.format(self.n_semantic_neighbors))

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not((isinstance(self.const_range, tuple) and
                len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.feature_names is not None:
            if self.n_features_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_, len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_val_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

            if self.edda_params is not None:
                # Adjust the log file for EDDA
                self.run_details_["deme"] = []
                self.run_details_["gp_deme"] = []

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.log:
            log_event = [self.random_state]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
        else:
            logger = None

        for gen in range(prior_generations, self.generations):

            start_time = time()

            if gen == 0:
                if self.edda_params is not None:
                    # Use EDDA initialization
                    parents = _initialize_edda(params, self.population_size, X, y, sample_weight, train_indices,
                                               val_indices, self.verbose, logger, random_state, self.n_jobs)
                else:
                    # Use standard initialization
                    parents = None

                if self.verbose:
                    # Print header fields for MEP
                    self._verbose_reporter()
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs, verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          train_indices,
                                          val_indices,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            # Semantic Stopping Criteria
            tie = 0.0 < self.tie_stopping_criteria <= 1.0
            edv = 0.0 < self.edv_stopping_criteria <= 1.0
            if tie or edv:
                seeds = random_state.randint(MAX_INT, size=self.n_semantic_neighbors)
                # params, seeds
                tie_, edv_ = _get_semantic_stopping_criteria(n_semantic_neighbors=self.n_semantic_neighbors,
                                                             elite=best_program,
                                                             X=X,
                                                             y=y,
                                                             sample_weight=sample_weight,
                                                             train_indices=train_indices,
                                                             params=params,
                                                             seeds=seeds)
                if tie and tie_ < self.tie_stopping_criteria:
                    print("Evolution stopped at generation {0:} "
                          "as TIE ({1:.2f}) < {2:.2f} threshold".format(gen, tie_, self.tie_stopping_criteria))
                    break

                if edv and edv_ < self.edv_stopping_criteria:
                    print("Evolution stopped at generation {0:} "
                          "as EDV ({1:.2f}) < {2:.2f} threshold".format(gen, edv_, self.edv_stopping_criteria))
                    break

            if self.edda_params is not None:
                self.run_details_['deme'].append("-1")
                self.run_details_['gp_deme'].append("-1")
            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            val_fitness = np.nan
            if self.val_set > 0.0:
                val_fitness = best_program.val_fitness_
            self.run_details_["best_val_fitness"].append(val_fitness)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            if self.log:
                log_event = [detailsList[-1] for detailsList in self.run_details_.values()]
                logger.info(','.join(list(map(str, log_event))))

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break

        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute(X) for gp in
                                   [self._programs[-1][i] for
                                    i in hall_of_fame]])
            if self.metric == 'spearman':
                evaluation = np.apply_along_axis(rankdata, 1, evaluation)

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = np.abs(np.corrcoef(evaluation))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(np.argmax(correlations),
                                                   correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i] for i in
                                   hall_of_fame[components]]

        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

        return self


class SymbolicRegressor(BaseSymbolic, RegressorMixin):

    """A Genetic Programming symbolic regressor.

    A symbolic regressor is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    metric : str, optional (default='mean absolute error')
        The name of the raw fitness metric. Available options include:

        - 'mean absolute error'.
        - 'mse' for mean squared error.
        - 'rmse' for root mean squared error.
        - 'pearson', for Pearson's product-moment correlation coefficient.
        - 'spearman' for Spearman's rank-order correlation coefficient.

        Note that 'pearson' and 'spearman' will not directly predict the target
        but could be useful as value-added features in a second-step estimator.
        This would allow the user to generate one engineered feature at a time,
        using the SymbolicTransformer would allow creation of multiple features
        at once.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicTransformer

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 tie_stopping_criteria=0.0,
                 edv_stopping_criteria=0.0,
                 n_semantic_neighbors=0,
                 const_range=(-1., 1.),
                 init_depth=(2, 4),
                 init_method='half and half',
                 edda_params=None,
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_simple_crossover=0.9,
                 p_p2_crossover=0.9,
                 p_uniform_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_shake_mutation=0.01,
                 p_graft_mutation=0.01,
                 p_reverse_mutation=0.01,
                 p_swap_mutation=0.01,
                 p_point_replace=0.05,
                 p_gs_crossover=0.0,
                 p_gs_mutation=0.0,
                 gsm_ms=-1.0,
                 semantical_computation=False,
                 val_set=0.0,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 log=False,
                 random_state=None):
        super(SymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            tie_stopping_criteria=tie_stopping_criteria,
            edv_stopping_criteria=edv_stopping_criteria,
            n_semantic_neighbors=n_semantic_neighbors,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            edda_params=edda_params,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_simple_crossover=p_simple_crossover,
            p_p2_crossover=p_p2_crossover,
            p_uniform_crossover=p_uniform_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_shake_mutation=p_shake_mutation,
            p_graft_mutation=p_graft_mutation,
            p_reverse_mutation=p_reverse_mutation,
            p_swap_mutation=p_swap_mutation,
            p_point_replace=p_point_replace,
            p_gs_crossover=p_gs_crossover,
            p_gs_mutation=p_gs_mutation,
            gsm_ms=gsm_ms,
            semantical_computation=semantical_computation,
            val_set=val_set,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            log=log,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(self, X):
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted values for X.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicRegressor not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_, n_features))

        y = self._program.execute(X)

        return y


class SymbolicClassifier(BaseSymbolic, ClassifierMixin):

    """A Genetic Programming symbolic classifier.

    A symbolic classifier is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional (default=500)
        The number of programs in each generation.

    generations : integer, optional (default=10)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    transformer : str, optional (default='sigmoid')
        The name of the function through which the raw decision function is
        passed. This function will transform the raw decision function into
        probabilities of each class.

        This can also be replaced by your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

    metric : str, optional (default='log loss')
        The name of the raw fitness metric. Available options include:

        - 'log loss' aka binary cross-entropy loss.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicTransformer

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer='sigmoid',
                 metric='log loss',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_simple_crossover=0.9,
                 p_p2_crossover=0.9,
                 p_uniform_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_shake_mutation=0.01,
                 p_graft_mutation=0.01,
                 p_reverse_mutation=0.01,
                 p_swap_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicClassifier, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            transformer=transformer,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_simple_crossover=p_simple_crossover,
            p_p2_crossover=p_p2_crossover,
            p_uniform_crossover=p_uniform_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_shake_mutation=p_shake_mutation,
            p_graft_mutation=p_graft_mutation,
            p_reverse_mutation=p_reverse_mutation,
            p_swap_mutation=p_swap_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict_proba(self, X):
        """Predict probabilities on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        proba : array, shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicClassifier not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_, n_features))

        scores = self._program.execute(X)
        proba = self._transformer(scores)
        proba = np.vstack([1 - proba, proba]).T
        return proba

    def predict(self, X):
        """Predict classes on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples,]
            The predicted classes of the input samples.

        """
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)


class SymbolicTransformer(BaseSymbolic, TransformerMixin):

    """A Genetic Programming symbolic transformer.

    A symbolic transformer is a supervised transformer that begins by building
    a population of naive random formulas to represent a relationship. The
    formulas are represented as tree-like structures with mathematical
    functions being recursively applied to variables and constants. Each
    successive generation of programs is then evolved from the one that came
    before it by selecting the fittest individuals from the population to
    undergo genetic operations such as crossover, mutation or reproduction.
    The final population is searched for the fittest individuals with the least
    correlation to one another.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    hall_of_fame : integer, or None, optional (default=100)
        The number of fittest programs to compare from when finding the
        least-correlated individuals for the n_components. If `None`, the
        entire final generation will be used.

    n_components : integer, or None, optional (default=10)
        The number of best programs to return after searching the hall_of_fame
        for the least-correlated individuals. If `None`, the entire
        hall_of_fame will be used.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=1.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    metric : str, optional (default='pearson')
        The name of the raw fitness metric. Available options include:

        - 'pearson', for Pearson's product-moment correlation coefficient.
        - 'spearman' for Spearman's rank-order correlation coefficient.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicRegressor

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 population_size=1000,
                 hall_of_fame=100,
                 n_components=10,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=1.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='pearson',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_simple_crossover=0.9,
                 p_p2_crossover=0.9,
                 p_uniform_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_shake_mutation=0.01,
                 p_graft_mutation=0.01,
                 p_reverse_mutation=0.01,
                 p_swap_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicTransformer, self).__init__(
            population_size=population_size,
            hall_of_fame=hall_of_fame,
            n_components=n_components,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_simple_crossover=p_simple_crossover,
            p_p2_crossover=p_p2_crossover,
            p_uniform_crossover=p_uniform_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_shake_mutation=p_shake_mutation,
            p_graft_mutation=p_graft_mutation,
            p_reverse_mutation=p_reverse_mutation,
            p_swap_mutation=p_swap_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __len__(self):
        """Overloads `len` output to be the number of fitted components."""
        if not hasattr(self, '_best_programs'):
            return 0
        return self.n_components

    def __getitem__(self, item):
        """Return the ith item of the fitted components."""
        if item >= len(self):
            raise IndexError
        return self._best_programs[item]

    def __str__(self):
        """Overloads `print` output of the object to resemble LISP trees."""
        if not hasattr(self, '_best_programs'):
            return self.__repr__()
        output = str([gp.__str__() for gp in self])
        return output.replace("',", ",\n").replace("'", "")

    def transform(self, X):
        """Transform X according to the fitted transformer.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_components]
            Transformed array.

        """
        if not hasattr(self, '_best_programs'):
            raise NotFittedError('SymbolicTransformer not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_, n_features))

        X_new = np.array([gp.execute(X) for gp in self._best_programs]).T

        return X_new

    def fit_transform(self, X, y, sample_weight=None):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_components]
            Transformed array.

        """
        return self.fit(X, y, sample_weight).transform(X)
