from skopt.space import Real, Integer, Categorical

params = {
    "hsppbo":
    [
        ('alpha', Integer(1, 10)),
        ('beta', Integer(1, 10)),
        #('w_rand', Real(0, 0.1)),
        #('w_pers_best', Real(0, 0.1)),
        #('w_pers_prev', Real(0, 0.1)),
        #('w_parent_best', Real(0, 0.2)),
        #('detection_threshold', Real(0, 0.5)),
        #('reaction_type', Categorical(['partial', 'full', 'none'])),
    ]
}
