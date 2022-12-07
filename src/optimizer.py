from typing import Any
import skopt
from functools import partial
from config import optimizer_conf


class Optimizer:
    def __init__(self, optimization_func, objective_func: str, params: list[tuple]) -> None:
        """
        Create an optimizer wrapper handling all of the available hyperoptimization functions

        Args:
            optimization_func (str): The chosen hyperoptimization functions
            objective_func: The function to optimize
            params (list[tuple[str, skopt.Dimension]]): List of tuples, specifying the parameters to optimize, described by the string param name and the dimension of the param
                                                        The provided objective function should take all of the provided params as input
        """
        name_to_func = {
            'random': random_search,
            'bayesian': bayesian_search,
            'forest': tree_regression,
            'gradient': gradient_boost_regression
        }
        self.optimization_func_name = optimization_func
        self.run = partial(name_to_func[self.optimization_func_name],
                           objective_func, params, optimizer_conf[self.optimization_func_name])


class ObjFunc:
    def __init__(self, objective_func, params: list[tuple]) -> None:
        """
        The wrapper for the objective function, that shall be optimized

        Args:
            objective_func: The function to optimize
            params (list[tuple[str, skopt.Dimension]]): List of tuples, specifying the parameters to optimize, described by the string param name and the dimension of the param.
                                                        The provided objective function should take all of the provided params as input
        """
        self.objective_func = objective_func
        self.dimensions = self.convert_params(params)

    def __call__(self, *args: Any) -> Any:
        return self.objective_func(*args)

    @staticmethod
    def convert_params(params: list[tuple]):
        """
        Convert the param list into a format for skopt

        Args:
            params (list[tuple[str, skopt.Dimension]]): List of tuples, specifying the parameters to optimize, described by the string param name and the dimension of the param.

        Returns:
            list[skopt.Dimension]: List of the to optimize parameters' dimensions 
        """
        return [x[1] for x in params]


def random_search(func, params, conf, verbose=False, random_state=None, n_calls=100):
    """
    Random search by uniform sampling within the given bounds.

    Args:
        func: Objective function to optimize
        params (list[tuple[str, skopt.Dimension]]): List of tuples, specifying the parameters to optimize, described by the string param name and the dimension of the param.
                                                        The provided objective function should take all of the provided params as input
        conf: Additional configuration for the optimizer
        verbose (bool, optional): Enable output during the optimizer runs. Defaults to False.
        random_state (optional): Random state to set the RNG of the optimizer. Defaults to None.
        n_calls (int, optional): Sets how many function calls/algorithm runs the optimizer should perform. Defaults to 100.

    Returns:
        scipy.OptimizeResult (modified by skopt):
        The optimization result returned as a OptimizeResult object.
        Important attributes are:

        - `x` [list]: location of the minimum.
        - `fun` [float]: function value at the minimum.
        - `x_iters` [list of lists]: location of function evaluation for each
          iteration.
        - `func_vals` [array]: function value for each iteration.
        - `space` [Space]: the optimisation space.
        - `specs` [dict]: the call specifications.
        - `rng` [RandomState instance]: State of the random state
          at the end of minimization.
    """
    f = ObjFunc(func, params)
    try:
        return skopt.dummy_minimize(func=f, dimensions=f.dimensions,
                                    n_calls=n_calls, random_state=random_state, verbose=verbose, **conf)
    except Exception as e:
        import traceback
        traceback.print_exc()


def bayesian_search(func, params, conf, verbose=False, random_state=None, n_calls=100):
    """
    Bayesian optimization using Gaussian Processes.

    Args:
        func: Objective function to optimize
        params (list[tuple[str, skopt.Dimension]]): List of tuples, specifying the parameters to optimize, described by the string param name and the dimension of the param.
                                                        The provided objective function should take all of the provided params as input
        conf: Additional configuration for the optimizer
        verbose (bool, optional): Enable output during the optimizer runs. Defaults to False.
        random_state (optional): Random state to set the RNG of the optimizer. Defaults to None.
        n_calls (int, optional): Sets how many function calls/algorithm runs the optimizer should perform. Defaults to 100.

    Returns:
        scipy.OptimizeResult (modified by skopt):
        The optimization result returned as a OptimizeResult object.
        Important attributes are:

        - `x` [list]: location of the minimum.
        - `fun` [float]: function value at the minimum.
        - `x_iters` [list of lists]: location of function evaluation for each
          iteration.
        - `func_vals` [array]: function value for each iteration.
        - `space` [Space]: the optimisation space.
        - `specs` [dict]: the call specifications.
        - `rng` [RandomState instance]: State of the random state
          at the end of minimization.
    """
    f = ObjFunc(func, params)
    try:
        return skopt.gp_minimize(func=f, dimensions=f.dimensions,
                                 n_calls=n_calls, random_state=random_state, verbose=verbose, **conf)
    except Exception as e:
        import traceback
        traceback.print_exc()


def tree_regression(func, params, conf, verbose=False, random_state=None, n_calls=100):
    """
    Sequential optimization using gradient boosted trees.

    Args:
        func: Objective function to optimize
        params (list[tuple[str, skopt.Dimension]]): List of tuples, specifying the parameters to optimize, described by the string param name and the dimension of the param.
                                                        The provided objective function should take all of the provided params as input
        conf: Additional configuration for the optimizer
        verbose (bool, optional): Enable output during the optimizer runs. Defaults to False.
        random_state (optional): Random state to set the RNG of the optimizer. Defaults to None.
        n_calls (int, optional): Sets how many function calls/algorithm runs the optimizer should perform. Defaults to 100.

    Returns:
        scipy.OptimizeResult (modified by skopt):
        The optimization result returned as a OptimizeResult object.
        Important attributes are:

        - `x` [list]: location of the minimum.
        - `fun` [float]: function value at the minimum.
        - `x_iters` [list of lists]: location of function evaluation for each
          iteration.
        - `func_vals` [array]: function value for each iteration.
        - `space` [Space]: the optimisation space.
        - `specs` [dict]: the call specifications.
        - `rng` [RandomState instance]: State of the random state
          at the end of minimization.
    """
    f = ObjFunc(func, params)
    try:
        return skopt.forest_minimize(func=f, dimensions=f.dimensions,
                                     n_calls=n_calls, random_state=random_state, verbose=verbose, **conf)
    except Exception as e:
        import traceback
        traceback.print_exc()


def gradient_boost_regression(func, params, conf, verbose=False, random_state=None, n_calls=100):
    """
    Sequential optimization using gradient boosted trees.

    Args:
        func: Objective function to optimize
        params (list[tuple[str, skopt.Dimension]]): List of tuples, specifying the parameters to optimize, described by the string param name and the dimension of the param.
                                                        The provided objective function should take all of the provided params as input
        conf: Additional configuration for the optimizer
        verbose (bool, optional): Enable output during the optimizer runs. Defaults to False.
        random_state (optional): Random state to set the RNG of the optimizer. Defaults to None.
        n_calls (int, optional): Sets how many function calls/algorithm runs the optimizer should perform. Defaults to 100.

    Returns:
        scipy.OptimizeResult (modified by skopt):
        The optimization result returned as a OptimizeResult object.
        Important attributes are:

        - `x` [list]: location of the minimum.
        - `fun` [float]: function value at the minimum.
        - `x_iters` [list of lists]: location of function evaluation for each
          iteration.
        - `func_vals` [array]: function value for each iteration.
        - `space` [Space]: the optimisation space.
        - `specs` [dict]: the call specifications.
        - `rng` [RandomState instance]: State of the random state
          at the end of minimization.
    """
    f = ObjFunc(func, params)
    try:
        return skopt.gbrt_minimize(func=f, dimensions=f.dimensions,
                                   n_calls=n_calls, random_state=random_state, verbose=verbose, **conf)
    except Exception as e:
        import traceback
        traceback.print_exc()
