from skopt.space import Real, Integer, Categorical

params = {
    "opt": {
        "hsppbo":
        [
            ('alpha', Integer(1, 10)),
            ('beta', Integer(1, 10)),
            #('w_rand', Real(0, 0.1)),
            ('w_pers_best', Real(0.01, 0.1)),
            ('w_pers_prev', Real(0.01, 0.1)),
            ('w_parent_best', Real(0.005, 0.02)),
            #('detection_threshold', Real(0, 0.5)),
            #('reaction_type', Categorical(['partial', 'full', 'none'])),
        ]},
    "exp": {
        "hsppbo":
        [
            ('detection_threshold', 0.1, 0.25, 0.5)
        ],
        "problem":
        [
            ('dynamic_intensity', 10, 25, 50),
        ]
    }
}

logger_run_params = {
    "hsppbo": ["iteration", "abs_runtime", "func_evals", "swaps", "reaction", "best_solution"]
}

optimizer_conf = {
    "bayesian": {
        'acq_func': 'PI',
        'xi': 0.01,
        'n_initial_points': 1,
        'initial_point_generator': 'hammersly',
        'noise': 0.07,
    },
    "random": {
        'initial_point_generator': 'random',
    },
    "forest": {},
    "gradient": {}
}
