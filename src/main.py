import argparse
from hsppbo import HSPPBO
from logger import Logger
from optimizer import Optimizer
from tsp import TSP
from config import params
import timeit


def user_input():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose',
                        help="Turn on output verbosity", action='store_true')
    parser.add_argument('-m', '--mode', type=str, choices=[
                        'run', 'opt', 'exp'], default='run', help='Mode of execution for the given problem and/or algorithm')
    parser.add_argument('-p', '--problem', type=str, default='rat195',
                        help='Name of the problem instance, e.g. the TSPLIB names like "rat195"')
    parser.add_argument('-pt', '--problem-type', type=str, default='TSP',
                        help='Type of the problem, e.g. TSP (standard, symmetric TSP), ATSP (asymmetric TSP), QAP')
    parser.add_argument('-di', '--dynamic-intensity', type=check_percent_range,
                        default=0.2, help='Instensity of the dynamic problem instance')
    parser.add_argument('-plb', '--personal-best', type=check_percent_range, default=0.075,
                        help='Influence of the SCE\'s best personal solution ever obtained')
    parser.add_argument('-ppr', '--personal-previous', type=check_percent_range,
                        default=0.075, help='Influence of the SCE\'s previous personal solution')
    parser.add_argument('-ptb', '--parent-best', type=check_percent_range, default=0.01,
                        help='Influence of the SCE\'s parent personal best solution ever obtained')
    parser.add_argument('-a', '--alpha', type=int, default=1,
                        help='Influence of the pheromone/non-heuristic')
    parser.add_argument('-b', '--beta', type=int, default=7,
                        help='Influence of the heuristic')
    parser.add_argument('-ddt', '--dynamic-detection-threshold', type=check_percent_range, default=0.25,
                        help='Threshold (swaps per SCE and iteration) for detecting a change')
    parser.add_argument('-r', '--reaction-type', type=str, choices=[
                        'partial', 'full', 'none'], default='partial', help='Type of reaction algorithm used to handle a change')

    args = parser.parse_args()

    return args


# check if number is between 0 and 1
def check_percent_range(number: float) -> float:
    try:
        number = float(number)
    except ValueError:
        raise argparse.ArgumentTypeError('Number is no floating point literal')

    if 0.0 > number or number > 1.0:
        raise argparse.ArgumentTypeError('Number has to be between 0 and 1')

    return number

# TODO:
#   - implement all logging cases (run: completed, opt: completed, exp: in progress )
#   - all modes: add relative difference to optimal solution to every output
#   - experiment mode: create csv for experiment, averaged over multiple runs
#   - implement analyzer/grapher module
#   - implement tests
#   - research sensible parameter boundaries for optimization
#   - web UI (flask) and better packaging


def main():
    args = user_input()

    logger = Logger(mode=args.mode)

    if (args.problem_type == 'TSP'):
        problem = TSP(tsplib_name=args.problem)
    else:
        raise NotImplementedError(
            'Problem type not implemented yet')

    # set wether the problem is dynamic or not
    if args.dynamic_intensity > 0:
        problem.set_dynamic(args.dynamic_intensity)

    hsppbo = HSPPBO(problem, logger, args.personal_best, args.personal_previous, args.parent_best,
                    args.alpha, args.beta, args.dynamic_detection_threshold, args.reaction_type, max_iteration_count=100)

    logger.create_info_log(hsppbo.get_info())

    match args.mode:
        case 'run':
            logger.init_mode()
            starttime = timeit.default_timer()
            solution = hsppbo.execute(verbose=args.verbose)
            print("Solution quality:", solution[1])
            print("Total execution time:", timeit.default_timer() - starttime)

        case 'exp':
            runs = get_run_number()
            logger.init_mode(runs)
            for i in range(1, runs+1):
                hsppbo.execute(verbose=args.verbose)
                hsppbo.tree.reset()
                logger.add_run_exp()
            logger.create_exp_avg_run()

        case 'opt':
            hsppbo.set_random_seed()
            runs = get_run_number()
            logger.init_mode(params['opt']['hsppbo'])
            opt = Optimizer("bayesian", hsppbo.execute_wrapper,
                            params['opt']['hsppbo'])
            for i in range(1, runs+1):
                print("---STARTING OPTIMIZATION RUN " +
                      str(i) + "/" + str(runs) + "---")
                opt_res = opt.run(verbose=args.verbose,
                                  n_calls=10, random_state=i)
                logger.create_opt_files(opt_res, run=i)
            logger.create_opt_best_params()


def get_run_number() -> None:
    print('Do you want to perform multiple runs? [Y/N]')
    x = input()
    if x.lower() == 'y' or x.lower() == 'yes':
        print('How many runs do you want to execute? (max. 30)')
        runs = input()
        if 0 < int(runs) <= 30:
            return int(runs)
    return 1


if __name__ == '__main__':
    main()
