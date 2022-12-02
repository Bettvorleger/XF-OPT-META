from problem import Problem
from functools import partial
from math import ceil
import tsplib95
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json


class TSP(Problem):
    dynamic = False

    # Fix rounding issue for eucledian 2D weight calc, always output float
    tsplib95.distances.TYPES["EUC_2D"] = partial(
        tsplib95.distances.euclidean, round=lambda x: x)

    def __init__(self, tsplib_name: str, problem_path="../problems/tsp/") -> None:
        """
        Creates a problem instance of a standard TSPLIB95 problem

        Args:
            tsplib_name (str): Name of a standard TSPLIB95 instance, e.g. 'rat195'
            problem_path (str, optional): Path to the problem instances. Defaults to "../problems/tsp/".
        """
        self.problem_path = problem_path
        self.instance = tsplib95.load(self.problem_path+tsplib_name+'.tsp')
        self._dimension = self.instance.dimension
        self.distance_matrix = self.get_distance_matrix()
        self.recipr_dist_matrix = np.reciprocal(
            self.distance_matrix, where=self.distance_matrix != 0)

        self.dynamic_intensity = int(ceil(0.2 * self.dimension))
        self.dynamic_frequency = 100
        self.min_iteration_count = 2000-1

    def set_dynamic(self, dynamic_intensity=0.2, dynamic_frequency=100, min_iteration_count=2000-1) -> None:
        """
        Apply dynamic properties to the TSP, making it a dynamic TSP

        Args:
            dynamic_intensity (float, optional): Percantage of how much is going to change relative to the cities. Defaults to 0.2.
            dynamic_frequency (int, optional): After how many iterations a change occurs. Defaults to 100.
        """
        self.dynamic_intensity = int(ceil(
            dynamic_intensity * self.dimension))  # How many cities are going to change per dynamic call

        # for dynamic TSP the cities are swaped in pairs, so the intensity needs to be even
        if self.dynamic_intensity % 2 != 0:
            self.dynamic_intensity -= 1

        # After how many iterations a change occurs
        self.dynamic_frequency = dynamic_frequency

        # Number of iterations before dynamic starts
        self.min_iteration_count = min_iteration_count

        self.rng = np.random

        self.dynamic = True

    def set_random_seed(self):
        """
        Fixing the seed of the RNG, making the results predictable
        """
        self.rng.seed(0)

    def init_solution(self) -> tuple[tuple[int], int]:
        """
        Initialize an new possible solution for the given TSP instance.
        Creates list/route of random indices with length equal to number of cities

        Returns:
            tuple[list[int], int]: a possible route for the given problem instance, and its solution quality (length)
        """
        solution = list(range(0, self.dimension))
        solution = self.rng.permutation(solution).tolist()
        return tuple(solution), self.get_solution_quality(solution)

    def check_dynamic_change(self, iteration_count: int):
        """
        Checks if dynamic change should happen in current iteration and triggers it if necessary

        Args:
            iteration_count (int): Current iteration of the algo
        """
        if self.dynamic:
            if iteration_count % self.dynamic_frequency == 0 and iteration_count > self.min_iteration_count:

                cities = self.rng.permutation(range(self.dimension))
                cities = cities[0:self.dynamic_intensity]

                city_pairs = [list(a) for a in np.array_split(
                    cities, len(cities)/2)]

                for pair in city_pairs:
                    self.swap_nodes(pair[0], pair[1])

                self.distance_matrix = self.get_distance_matrix()
                self.recipr_dist_matrix = np.reciprocal(
                    self.distance_matrix, where=self.distance_matrix != 0)
                return True
        return False

    def get_distance_matrix(self) -> np.ndarray:
        """
        Get the distance matrix of the TSPLIB instance

        Returns:
            np.ndarray: ND-array with the problem instace equivalent distance matrix
        """
        distance_matrix_flattended = np.array([
            self.instance.get_weight(*edge) for edge in self.instance.get_edges()
        ])
        distance_matrix = np.reshape(
            distance_matrix_flattended, (self.dimension,
                                         self.dimension)
        )
        # Fill diagonals with 0
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix

    def get_solution_quality(self, path: tuple[int]) -> float:
        """
        Get the solution quality, which, for TSP, is the length of the given path.
        Also closing the tour via last node to first node.

        Args:
            path (tuple[int]): Path as possible solution

        Returns:
            int: Length of the tour
        """
        length = 0
        for idn, node in enumerate(path):
            next_node = path[idn + 1] if idn + 1 < self.dimension else path[0]
            length += self.distance_matrix[node][next_node]
        return length

    def get_heuristic_component(self, i: int, j: int) -> float:
        return self.recipr_dist_matrix[i][j]

    def get_optimal_solution(self) -> float:
        try:
            with open(self.problem_path+'../optimal.json', 'r') as f:
                optimal = json.load(f)
                return optimal["tsp"]["tsplib"]["stsp"][self.instance.name]
        except:
            return None

    def get_average_distance(self) -> float:
        """
        Calculate the average distance of the tsp paths via the distance matrix

        Returns:
            float: average distance between nodes
        """
        return np.average(self.distance_matrix)

    def get_median_distance(self) -> float:
        """
        Calculate the median distance of the tsp paths via the distance matrix

        Returns:
            float: median distance between nodes
        """
        return np.median(self.distance_matrix)

    def swap_nodes(self, node_1: int, node_2: int):
        """
        Swap two given node indices in the problem instance

        Args:
            node_1 (int): Node Index of first city
            node_2 (int): Node Index of second city
        """

        # add one to each index, because its not a list, but an IndexedCoordinatesField starting with 1
        node_1 += 1
        node_2 += 1

        if (node_1 <= self.dimension and node_2 <= self.dimension):
            self.instance.node_coords[node_1], self.instance.node_coords[
                node_2] = self.instance.node_coords[node_2], self.instance.node_coords[node_1]
        else:
            raise ValueError(
                'Node indices out of bounds. Needs to be between 0 and '+str(self.dimension-1)+', but values '+str(node_1-1)+' and '+str(node_2-1)+' were given.')

    def visualize(self, solution=None, interactive=True, filepath=".",) -> None:
        """
        Create an interactive view or png image from the problem instance and optionally, the solution path

        Args:
            interactive (bool, optional): Wether the interactive plot viewer should be used. Defaults to True.
            path (str, optional): If non-interactive, which relative path the image is saved to. Defaults to ".".
        """
        G = self.instance.get_graph()

        # get the coords for each node and the edgelist (pairwise combination of each solution element with its next node)
        pos = nx.get_node_attributes(G, "coord")

        options = {
            "font_size": 8,
            "node_size": 250,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 0.5,
            "width": 0,
        }
        nx.draw_networkx(
            G,
            pos,
            **options)

        if solution is not None:
            # since networkx and the TSPLIB start their node count at 1, not 0, we add 1 to every element in the solution path
            solution = [x+1 for x in solution]
            el = list(nx.utils.pairwise(solution, True))
            nx.draw_networkx_edges(G, pos, edgelist=el,
                                   edge_color="red", width=2)

        plt.axis("off")
        ax = plt.gca()
        ax.margins(0)
        ax.set_axis_off()
        plt.subplots_adjust(bottom=0, right=1, top=1, left=0)
        if interactive:
            plt.show()
        else:
            plt.savefig(filepath+"/"+self.instance.name +
                        ".png", bbox_inches="tight", )

    @property
    def dimension(self) -> int:
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        self._dimension = value

    def get_info(self) -> dict:
        """
        Get information about the current TSP instance as a dict

        Returns:
            dict: Information about the TSP instance
        """
        dynamic_props = {}
        if self.dynamic:
            dynamic_props = {
                "dynamic_frequency": self.dynamic_frequency,
                "dynamic_intensity": self.dynamic_intensity,
                "min_iterations_before_dynamic": self.min_iteration_count
            }
        return {
            "type": "stsp",
            "name": self.instance.name,
            "dimension": self.dimension,
            "weight_type": self.instance.edge_weight_type,
            "dynamic_props": dynamic_props
        }
    
    def __str__(self) -> str:
        dynamic_props = ""
        if self.dynamic:
            dynamic_props = "Dynamic Frequency: %s, Dynamic Intensity: %d, Minimum Iterations before Dynamic: %s\n" % (
                self.dynamic_frequency, self.dynamic_intensity, self.min_iteration_count)
        return "\nType: Symmetric TSP, Name: %s, Dimension: %s, Weight Type: %s\n" % (self.instance.name, self.dimension, self.instance.edge_weight_type) + dynamic_props

    __repr__ = __str__
