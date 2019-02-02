import warnings

import numpy as np
import sklearn

import experiments
import learners


class SVMExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/SVM.py
        samples = self._details.ds.features.shape[0]
        features = self._details.ds.features.shape[1]

        gamma_fracs = np.arange(1 / features, 2.1, 0.2)
        tols = np.arange(1e-8, 1e-1, 0.01)
        C_values = np.arange(0.001, 2.5, 0.25)
        iters = [-1, int((1e6 / samples) / .8) + 1]

        best_params_linear = None
        best_params_rbf = None
        # Uncomment to select known best params from grid search. This will skip the grid search and just rebuild
        # the various graphs
        #
        # Dataset 1:
        # best_params_linear = {'C': 0.5, 'class_weight': 'balanced', 'loss': 'squared_hinge',
        #                       'max_iter': 1478, 'tol': 0.06000001}
        # best_params_rbf = {'C': 2.0, 'class_weight': 'balanced', 'decision_function_shape': 'ovo',
        #                    'gamma': 0.05555555555555555, 'max_iter': -1, 'tol': 1e-08}
        # Dataset 2:
        # best_params_linear = {'C': 1.0, 'class_weight': 'balanced', 'loss': 'hinge', 'dual': True,
        #                       'max_iter': 70, 'tol': 0.08000001}
        # best_params_rbf = {'C': 1.5, 'class_weight': 'balanced', 'decision_function_shape': 'ovo',
        #                    'gamma': 0.125, 'max_iter': -1, 'tol': 0.07000001}

        # Adult:
        # best_params_linear = {
        #     'C': 0.251,
        #     'cache_size': 200,
        #     'class_weight': 'balanced',
        #     'coef0': 0.0,
        #     'decision_function_shape': 'ovo',
        #     'degree': 3,
        #     'gamma': 0.017857142857142856,
        #     'kernel': 'linear',
        #     'max_iter': -1,
        #     'probability': False,
        #     'shrinking': True,
        #     'tol': 0.08000001,
        #     'verbose': False
        # }
        #
        # best_params_rbf = {
        #     'C': 1.251,
        #     'cache_size': 200,
        #     'class_weight': 'balanced',
        #     'coef0': 0.0,
        #     'decision_function_shape': 'ovo',
        #     'degree': 3,
        #     'gamma': 0.017857142857142856,
        #     'kernel': 'rbf',
        #     'max_iter': -1,
        #     'probability': False,
        #     'shrinking': True,
        #     'tol': 0.06000001,
        #     'verbose': False
        # }
        #
        # Spam:
        #
        # best_params_linear = {
        #     'C': 2.251,
        #     'cache_size': 200,
        #     'class_weight': 'balanced',
        #     'coef0': 0.0,
        #     'decision_function_shape': 'ovo',
        #     'degree': 3,
        #     'gamma': 0.017543859649122806,
        #     'kernel': 'linear',
        #     'max_iter': 1,
        #     'probability': False,
        #     'shrinking': True,
        #     'tol': 0.09000000999999999,
        #     'verbose': False
        # }

        # best_params_rbf = {
        #     'C': 2.001,
        #     'cache_size': 200,
        #     'class_weight': 'balanced',
        #     'coef0': 0.0,
        #     'decision_function_shape': 'ovo',
        #     'degree': 3,
        #     'gamma': 0.017543859649122806,
        #     'kernel': 'rbf',
        #     'max_iter': 1,
        #     'probability': False,
        #     'shrinking': True,
        #     'tol': 0.03000001,
        #     'verbose': False
        # }

        ########## RBF SVM
        params = {'SVM__max_iter': iters, 'SVM__tol': tols, 'SVM__class_weight': ['balanced'],
                  'SVM__C': C_values,
                  'SVM__decision_function_shape': ['ovo', 'ovr'], 'SVM__gamma': gamma_fracs}
        # complexity_param = {'name': 'SVM__C', 'display_name': 'Penalty', 'values': np.arange(0.001, 2.5, 0.1)}
        complexity_param = {'name': 'SVM__gamma', 'display_name': 'Gamma', 'values': np.arange(1 / features, 2.1, 0.2)}

        iteration_details = {
            # 'x_scale': 'log',
            'params': {'SVM__max_iter': [2 ** x for x in range(12)]},
        }

        learner = learners.SVMLearner(kernel='rbf')
        if best_params_rbf is not None:
            learner.set_params(**best_params_rbf)

        best_params = experiments.perform_experiment(
            self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner, 'SVM_RBF', 'SVM',
            params, complexity_param=complexity_param, seed=self._details.seed, iteration_details=iteration_details,
            best_params=best_params_rbf,
            threads=self._details.threads, verbose=self._verbose)

        # Overfitting - currently doesn't seem to do anything useful...

        # of_params = best_params.copy()
        # learner = learners.SVMLearner(kernel='rbf')
        # if best_params_rbf is not None:
        #     learner.set_params(**best_params_rbf)
        # experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner,
        #                                'SVM_RBF_OF', 'SVM', of_params, seed=self._details.seed,
        #                                iteration_details=iteration_details,
        #                                best_params=best_params_rbf,
        #                                threads=self._details.threads, verbose=self._verbose,
        #                                iteration_lc_only=True)

        ########## LINEAR SVM
        learner = learners.SVMLearner(kernel='linear')
        if best_params_linear is not None:
            learner.set_params(**best_params_linear)

        best_params = experiments.perform_experiment(
            self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner, 'SVM_LINEAR', 'SVM',
            params, complexity_param=complexity_param, seed=self._details.seed, iteration_details=iteration_details,
            best_params=best_params_rbf,
            threads=self._details.threads, verbose=self._verbose)

        # of_params = best_params.copy()
        # learner = learners.SVMLearner(kernel='linear')
        # if best_params_rbf is not None:
        #     learner.set_params(**best_params_rbf)
        # experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner,
        #                                'SVM_LINEAR_OF', 'SVM', of_params, seed=self._details.seed,
        #                                iteration_details=iteration_details,
        #                                best_params=best_params_rbf,
        #                                threads=self._details.threads, verbose=self._verbose,
        #                                iteration_lc_only=True)
