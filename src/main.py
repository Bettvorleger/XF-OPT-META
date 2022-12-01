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
                        'run', 'optimize', 'experiment'], default='run', help='Mode of execution for the given problem and/or algorithm')
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


def main():
    args = user_input()
    match args.mode:
        case 'run':
            logger = Logger(mode='run')
        case 'experiment':
            logger = Logger(mode='exp')
        case 'optimize':
            logger = Logger(mode='opt')

    if (args.problem_type == 'TSP'):
        problem = TSP(tsplib_name=args.problem)
    else:
        raise NotImplementedError(
            'Problem type not implemented yet')

    # set wether the problem is dynamic or not
    if args.dynamic_intensity > 0:
        problem.set_dynamic(args.dynamic_intensity)

    hsppbo = HSPPBO(problem, logger, args.personal_best, args.personal_previous, args.parent_best,
                    args.alpha, args.beta, args.dynamic_detection_threshold, args.reaction_type)

    match args.mode:
        case 'run':
            starttime = timeit.default_timer()
            if args.verbose:
                solution = hsppbo.execute(verbose=True)
            else:
                solution = hsppbo.execute()
            print("Solution quality:", solution[1])
            print("Total execution time:", timeit.default_timer() - starttime)
        case 'experiment':
            pass
        case 'optimize':
            hsppbo.set_random_seed()
            opt = Optimizer("bayesian", hsppbo.execute_wrapper, params['hsppbo'])
            opt.run(verbose=args.verbose)


if __name__ == '__main__':
    main()
