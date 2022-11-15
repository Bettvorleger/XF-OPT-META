from typing import Any
from skopt.space import Real, Integer, Categorical, Space
from skopt.utils import use_named_args
from skopt import gp_minimize
from functools import partial


class Optimizer:

    def __init__(self, optimization_func, objective_func) -> None:
        name_to_func = {
            'random': bayesian_search,
            'bayesian': bayesian_search,
        }
        self.optimization_func_name = optimization_func
        self.run = partial(name_to_func[self.optimization_func_name],
                           objective_func)


class ObjFunc:
    def __init__(self, objective_func, dimensions) -> None:
        self.objective_func = objective_func
        self.dimensions = dimensions

    def __call__(self, *args: Any) -> Any:
        return self.objective_func(*args)


def bayesian_search(func, verbose=False):
    conf = {
        'acq_func': 'PI',
        'xi': 0.01,
        'n_initial_points': 25,
        'initial_point_generator': 'hammersly',
        'noise': 0.07,
    }
    dimensions = [Integer(1, 10),
                  Integer(1, 10)]

    f = ObjFunc(func, dimensions)

    try:
        res_gp = gp_minimize(func=f, dimensions=f.dimensions,
                             n_calls=10, random_state=None, verbose=verbose)
        print(res_gp)
    except Exception as e:
        import traceback
        traceback.print_exc()
