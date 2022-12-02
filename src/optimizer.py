from typing import Any
import skopt
from functools import partial
from config import conf


class Optimizer:
    def __init__(self, optimization_func, objective_func, params) -> None:
        name_to_func = {
            'random': random_search,
            'bayesian': bayesian_search,
            'forest': tree_regression,
            'gradient': gradient_boost_regression
        }
        self.optimization_func_name = optimization_func
        self.run = partial(name_to_func[self.optimization_func_name],
                           objective_func, params, conf[self.optimization_func_name])


class ObjFunc:
    def __init__(self, objective_func, params) -> None:
        self.objective_func = objective_func
        self.dimensions = self.convert_params(params)

    def __call__(self, *args: Any) -> Any:
        return self.objective_func(*args)

    @staticmethod
    def convert_params(params):
        return [x[1] for x in params]


def random_search(func, params, conf, verbose=False, random_state=None, n_calls=100):
    f = ObjFunc(func, params)
    try:
        return skopt.dummy_minimize(func=f, dimensions=f.dimensions,
                                    n_calls=n_calls, random_state=random_state, verbose=verbose, **conf)
    except Exception as e:
        import traceback
        traceback.print_exc()


def bayesian_search(func, params, conf, verbose=False, random_state=None, n_calls=100):
    f = ObjFunc(func, params)
    try:
        return skopt.gp_minimize(func=f, dimensions=f.dimensions,
                                 n_calls=n_calls, random_state=random_state, verbose=verbose, **conf)
    except Exception as e:
        import traceback
        traceback.print_exc()


def tree_regression(func, params, conf, verbose=False, random_state=None, n_calls=100):
    f = ObjFunc(func, params)
    try:
        return skopt.forest_minimize(func=f, dimensions=f.dimensions,
                                     n_calls=n_calls, random_state=random_state, verbose=verbose, **conf)
    except Exception as e:
        import traceback
        traceback.print_exc()


def gradient_boost_regression(func, params, conf, verbose=False, random_state=None, n_calls=100):
    f = ObjFunc(func, params)
    try:
        return skopt.gbrt_minimize(func=f, dimensions=f.dimensions,
                                   n_calls=n_calls, random_state=random_state, verbose=verbose, **conf)
    except Exception as e:
        import traceback
        traceback.print_exc()
