from skopt.space import Real, Integer, Categorical

params = {
    "opt": {
        "hsppbo":
        [
            ('alpha', Integer(0, 10)),
            ('beta', Integer(0, 10)),
            ('w_pers_best', Real(0.001, 0.99)),
            ('w_pers_prev', Real(0.001, 0.99)),
            ('w_parent_best', Real(0.001, 0.99)),
            ('detection_threshold', Real(0.1, 0.5)),
            ('reaction_type', Categorical(['partial', 'full'])),
        ],
        "problem":
        [
            ('dynamic_intensity', 0.1, 0.25, 0.5),
        ]},
    "exp": {
        "hsppbo":
        [
            #('detection_threshold', 0.1, 0.25, 0.5)
        ],
        "problem":
        [
            ('dynamic_intensity', 0.1, 0.25, 0.5),
        ]
    }
}

# specify mode constants
MODE_RUN = "run"
MODE_EXPERIMENT = "exp"
MODE_OPTIMIZE = "opt"

output_folder_prefix = {MODE_RUN: "run_",
                        MODE_EXPERIMENT: "exp_", MODE_OPTIMIZE: "opt_"}

logger_run_params = {
    "hsppbo": ["iteration", "abs_runtime", "func_evals", "swaps", "reaction", "best_solution"]
}

optimizer_conf = {
    "bayesian": {
        'acq_func': 'PI',
        'n_initial_points': 10,
        'initial_point_generator': 'hammersly',
        'noise': 0.07,
    },
    "random": {
        'initial_point_generator': 'random',
    },
    "forest": {
        'base_estimator': 'ET'
    },
    "gradient": {}
}
