import numpy as np
import random
from os import urandom
from mpire import WorkerPool
from math import pow
from sce_tree import SCETree
from problem import Problem
from logger import Logger
from functools import partial
from config import params


class HSPPBO:
    max_iteration_count = 100  # maximum number of iterations
    sce_count = 13  # number of solution creating entities (SCE)
    sce_child_count = 3  # number of child nodes for each SCE
    # min number of iterations between change handling procedures (variable L in literaute)
    CHANGE_PAUSE = 5

    def __init__(self, problem: Problem, logger: Logger, pers_best=0.075, pers_prev=0.075, parent_best=0.01, alpha=1, beta=9, detection_threshold=0.25, reaction_type='partly') -> None:
        """
        Initialize the hsppbo algorithm with the given parameters.

        Args:
            problem (Problem): The problem instance that the algo is searching solutions for
            logger (Logger): The logger used during the algorithm run
            pers_best (float, optional): Influence of the SCE\'s best personal solution ever obtained. Defaults to 0.075.
            pers_prev (float, optional): Influence of the SCE\'s previous personal solution. Defaults to 0.075.
            parent_best (float, optional): Influence of the SCE\'s parent personal best solution ever obtained. Defaults to 0.01.
            alpha (int, optional): Influence of the probabilistic/non-heuristic component. Defaults to 1.
            beta (int, optional): Influence of the heuristic component. Defaults to 5.
            detection_threshold (float, optional): Threshold (swaps per SCE and iteration) for detecting a change. Defaults to 0.25.
            reaction_type (str, optional): Type of reaction algorithm used to handle a change in the problem. Defaults to 'partial'.
        """
        self.problem = problem
        self.logger = logger
        self.tree = SCETree(self.problem.init_solution,
                            self.sce_count, self.sce_child_count)

        self.w_pers_best = pers_best
        self.w_pers_prev = pers_prev
        self.w_parent_best = parent_best
        self.w_rand = 1 / (self.problem.dimension-1)
        self.alpha = alpha
        self.beta = beta
        self.detection_threshold = detection_threshold
        self.reaction_type = reaction_type

        self.fixed_rng = False

    def set_random_seed(self):
        """
        Fixing the seed of the RNG, making the results predictable
        """
        random.seed(0)
        self.problem.set_random_seed()
        self.fixed_rng = True

    def execute_wrapper(self, *args) -> int:
        """
        Args:
            *args (optional): 

        Returns:
            int: Best solution (tuple of path and length) found during the runtime of the algorithm.  
        """

        for k, v in enumerate(args[0]):
            self.__dict__[params['hsppbo'][k][0]] = v

        self.tree.reset()
        i = self.execute()[1]
        return i

    def execute(self, verbose=False) -> tuple[list, int]:
        """
        Excecute the HSPPBO algorithm

        Args:
            verbose (bool, optional): Show verbose outputs during and after the algorithm runs. Defaults to False.

        Returns:
            tuple[list, int]: Best solution (tuple of path and length) found during the runtime of the algorithm.       
        """
        self.tree.init_tree()
        change_pause_count = self.CHANGE_PAUSE

        for i in range(0, self.max_iteration_count):

            # check for dynamic change within the problem instance
            # if dynamic is triggered (true), recalculate the solution quality for each node
            if self.problem.check_dynamic_change(i):
                for sce in range(0, self.sce_count):
                    pp_quality = self.problem.get_solution_quality(
                        self.tree.get_solution(sce)[0])
                    pb_quality = self.problem.get_solution_quality(
                        self.tree.get_best_solution(sce)[0])
                    self.tree.set_node_quality(sce, pp_quality, pb_quality)

            # count up the change pause counter up to the constant
            if change_pause_count < self.CHANGE_PAUSE:
                change_pause_count += 1

            if (self.problem.dimension > 100):
                with WorkerPool() as pool:
                    # call the same function with different data in parallel, pass the current iteration as partial function
                    for sce, solution in pool.map_unordered(partial(self.construct_solution, iteration=i), range(0, self.sce_count)):
                        solution_quality = self.problem.get_solution_quality(
                            solution)
                        self.tree.update_node(sce, solution, solution_quality)
            else:
                # NON-MULTITHREADED VERSION
                for sce in range(0, self.sce_count):
                    solution = self.construct_solution(sce, i)[1]
                    solution_quality = self.problem.get_solution_quality(
                        solution)
                    self.tree.update_node(sce, solution, solution_quality)

            swap_count = 0  # number of swaps performed in the SCE tree
            for sce in self.tree.all_nodes_top_down:
                # get the pers best solution of the sce and its parent
                sce_solution_quality = self.tree.get_best_solution(
                    sce)[1]
                # get the solution quality of the sce and its parent
                parent_sce = self.tree.get_parent(sce)
                parent_solution_quality = self.tree.get_best_solution(
                    parent_sce)[1]
                # swap parent and sce position, if sce solution is better
                if sce_solution_quality < parent_solution_quality:
                    self.tree.swap_nodes(sce, parent_sce)
                    swap_count += 1

            # if the threshold for detecting a dynamic change is met,
            # trigger the change handling procedure of the tree and reset the change pause counter
            if swap_count > (self.detection_threshold * self.sce_count):
                if change_pause_count == self.CHANGE_PAUSE:
                    change_pause_count = 0
                    self.tree.change_handling(self.reaction_type)

            if i % 10 == 0 and verbose:
                print("Iteration: ", i, "\n")
                self.tree.tree.show(data_property="pb_quality")

            best_solution = self.tree.get_best_solution(self.tree.tree.root)[1]
            self.logger.log_iteration(
                i, self.sce_count*(i+1), swap_count, change_pause_count == 0, best_solution)

        best_solution = self.tree.get_best_solution(self.tree.tree.root)

        if verbose:
            self.tree.tree.show(data_property="pb_quality")
            self.problem.visualize(solution=best_solution[0])

        return best_solution

    def construct_solution(self, sce_index: int, iteration: int) -> tuple[int, tuple[int]]:
        """
        Constructs the solution path (tuple of node indices) accorinding to the algorithm of the HSPPBO paper

        Args:
            sce_index (int): Index of the SCE
            iteration(int): Current iteration, needed for fixed RNG

        Returns:
            tuple[int, tuple[int]]: A tuple containing the current SCE index (for multiprocessing reasons) and the solution path
        """

        if self.fixed_rng:
            random.seed(sce_index*iteration+iteration)
        else:
            random.seed(int.from_bytes(urandom(4), byteorder='little'))
        start_node = random.randrange(0, self.problem.dimension)

        # create list of unvisited nodes, remove first random node (called set U in paper)
        solution, unvisited = [start_node], list(
            range(0, self.problem.dimension))
        unvisited.remove(start_node)

        # get the solution populations from the tree as dict
        populations = self.tree.get_populations(sce_index)

        # first node already present, therefore subtract 1
        for i in range(0, self.problem.dimension-1):

            # create array of tau values for each unvisited path i,k with k as all unvisited nodes
            tau_arr = np.array(
                [self.tau(populations, solution[i], k) for k in tuple(unvisited)])

            # create probabilty distribution from the tau array and select a new node based on that
            rnd_distr = tau_arr / np.sum(tau_arr)
            next_node = random.choices(unvisited, rnd_distr)[0]

            # add new node to the solution path and remove it from the unvisited list
            solution.append(next_node)
            unvisited.remove(next_node)
        return sce_index, tuple(solution)

    def tau(self, populations: tuple, i: int, k: int) -> float:
        """
        Calculuates tau_(ik) given the population (solution paths), and taking into account the heurisitc, to the power of alpha and beta respectively.

        Args:
            populations (tuple): the populations (aka solution paths) of a SCE node
            i (int): the current node/state of the SCE
            k (int): the potential next node/state of the SCE

        Returns:
            float: the tau value for the given i,k
        """
        pop_range_sum = (self.w_pers_prev if self.is_solution_subset((i, k), populations[0]) else 0) + (self.w_pers_best if self.is_solution_subset(
            (i, k), populations[1]) else 0) + (self.w_parent_best if self.is_solution_subset((i, k), populations[2]) else 0)

        return pow(self.w_rand + pop_range_sum, self.alpha) * pow(self.problem.get_heuristic_component(i, k), self.beta)

    @staticmethod
    def is_solution_subset(subset: tuple, solution: dict) -> bool:
        """
        Checks wether the provided subset is a an ordered and concurrent part of the provided solution.
        That means gaps within the match of the subset are not allowed.
        Keep in mind, that the solution has to be a dictionary, with the nodes as key and index in the list as value, so each entry is (node: index) 

        Reason: Dicts can be searched in O(1), compared to lists/tuples in O(n)

        Example: 
        solution    subset
        [1,2,3,4]   [1,2]  -> TRUE
        [1,2,3,4]   [1,3]  -> FALSE

        Args:
            subset (tuple[int]): List of the subset to be matched for
            solution (tuple[int]): List of the solution to be matched against

        Returns:
            bool: True if the subset is contained within the solution as stated
        """
        try:
            return solution.get(subset[0]) + 1 == solution.get(subset[1])
        except:
            return False

    def get_info(self) -> dict:
        """
        Get information about the hsppbo parameters and its problem instace and SCE tree

        Returns:
            dict: info about the current hsppbo run
        """
        return {
            'hsppbo': {
                'w_pers_best': self.w_pers_best,
                'w_pers_prev': self.w_pers_prev,
                'w_parent_best': self.w_parent_best,
                'w_rand': self.w_rand,
                'alpha': self.alpha,
                'beta': self.beta,
                'detection_threshold': self.detection_threshold,
                'reaction_type': self.reaction_type,
                'fixed_rng': self.fixed_rng
            },
            'tree': self.tree.get_info(),
            'problem': self.problem.get_info()
        }
