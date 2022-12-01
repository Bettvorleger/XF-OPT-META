from typing import Any
from skopt import gp_minimize
from functools import partial


class Optimizer:
    def __init__(self, optimization_func, objective_func, params) -> None:
        name_to_func = {
            'random': bayesian_search,
            'bayesian': bayesian_search,
        }
        self.optimization_func_name = optimization_func
        self.run = partial(name_to_func[self.optimization_func_name],
                           objective_func, params)


class ObjFunc:
    def __init__(self, objective_func, params) -> None:
        self.objective_func = objective_func
        self.dimensions = self.convert_params(params)
        print(self.dimensions)

    def __call__(self, *args: Any) -> Any:
        return self.objective_func(*args)

    @staticmethod
    def convert_params(params):
        return [x[1] for x in params]




def bayesian_search(func, dimensions, verbose=False):
    conf = {
        'acq_func': 'PI',
        'xi': 0.01,
        'n_initial_points': 25,
        'initial_point_generator': 'hammersly',
        'noise': 0.07,
    }

    f = ObjFunc(func, dimensions)

    try:
        res_gp = gp_minimize(func=f, dimensions=f.dimensions,
                             n_calls=10, random_state=None, verbose=verbose)
        print(res_gp)
    except Exception as e:
        import traceback
        traceback.print_exc()
