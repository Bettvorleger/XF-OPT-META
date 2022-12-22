from problem import Problem
from functools import partial
from math import ceil
import tsplib95
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from scipy.spatial import ConvexHull


class TSP(Problem):
    dynamic = False
    TYPE = 'TSP'

    # Fix rounding issue for eucledian 2D weight calc, always output float
    tsplib95.distances.TYPES["EUC_2D"] = partial(
        tsplib95.distances.euclidean, round=lambda x: x)
    tsplib95.distances.TYPES["GEO"] = partial(
        tsplib95.distances.euclidean, round=lambda x: x)
    tsplib95.distances.TYPES["ATT"] = partial(
        tsplib95.distances.euclidean, round=lambda x: x)

    def __init__(self, tsplib_name: str, problem_path="../problems/tsp/") -> None:
        """
        Creates a problem instance of a standard TSPLIB95 problem

        Args:
            tsplib_name (str): Name of a standard TSPLIB95 instance, e.g. 'rat195'
            problem_path (str, optional): Path to the problem instances. Defaults to "../problems/tsp/".
        """
        self.tsplib_name = tsplib_name
        self.problem_path = problem_path
        self.instance = tsplib95.load(self.problem_path+tsplib_name+'.tsp')
        self._dimension = self.instance.dimension
        self.distance_matrix = self.get_distance_matrix()
        self.recipr_dist_matrix = np.reciprocal(
            self.distance_matrix, where=self.distance_matrix != 0)

    def reset(self) -> None:
        """
        Reset the problem to its original state and recalculate the distance matrix
        """
        self.instance = tsplib95.load(self.problem_path+self.tsplib_name+'.tsp')
        self.distance_matrix = self.get_distance_matrix()
        self.recipr_dist_matrix = np.reciprocal(
            self.distance_matrix, where=self.distance_matrix != 0)

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

                # get random permuatation of all city nodes
                cities = self.rng.permutation(range(self.dimension))
                # only use the amount specified by dynamic intensity
                cities = cities[0:self.dynamic_intensity]

                # create pairs with the split of the array in half
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
            with open(self.problem_path+'../metadata.json', 'r') as f:
                metadata = json.load(f)
                return metadata["tsp"]["tsplib"]["stsp"][self.instance.name]['optimal']
        except:
            return None

    def get_mean_distance(self) -> float:
        """
        Calculate the mean distance of the tsp paths via the distance matrix

        Returns:
            float: mean distance between nodes
        """
        return np.mean(self.distance_matrix)

    def get_median_distance(self) -> float:
        """
        Calculate the median distance of the tsp paths via the distance matrix

        Returns:
            float: median distance between nodes
        """
        return np.median(self.distance_matrix)

    def get_standard_deviation(self) -> float:
        """
        Calculate the standard deviation of distances of the tsp paths via the distance matrix

        Returns:
            float: standard deviation of distances between nodes
        """
        return np.std(self.distance_matrix)

    def get_coefficient_of_variation(self) -> float:
        """
        Calculate the coefficient of variation of distances of the tsp paths via the distance matrix

        Returns:
            float: coefficient of variation of distances between nodes
        """
        return (np.std(self.distance_matrix)/np.mean(self.distance_matrix))

    def get_first_eigenvalue(self) -> float:
        """
        Calculate the first eigenvalue of the distance matrix

        Returns:
            float: the first eigenvalue of the distance matrix
        """
        return (np.linalg.eig(self.distance_matrix)[0])[0].real

    def get_quartile_dispersion_coefficient(self) -> tuple[float, float]:
        """
        Calculate the quartile coefficient of dispersion via the distance matrix.
        It can be used as a robust version of the coefficient of variation.
        There are two definitions for the QDC:
        1. (Q3-Q1)/(Q3+Q1)
        2. (Q3-Q1)/Q2
        Version 2 is implemented

        Returns:
            tuple[float, float]: quartile coefficient of dispersion of distances between nodes
        """
        Q1 = np.quantile(self.distance_matrix, 0.25)
        Q3 = np.quantile(self.distance_matrix, 0.75)
        # return (Q3-Q1)/np.median(self.distance_matrix)
        return (Q3-Q1)/(Q3+Q1)

    def get_regularity_index(self) -> float:
        """
        Calculating the regularity index according to [1] and [2].
        It gives an impression of how much the distribution of nodes compares to a random one.
        A value of R = 1 would suggest a random distribution, values near 0 indicate a clustered distribution,
        and values greater than 1 point to a uniform distribution, with R = 2.1491 being the theoretical maximum of a triangular lattice arrangement according to [3].
        The generated trilattice64 arrangement, a two-dimensional grid graph of size 8x8 in which each square unit has a diagonal edge (provided within the visulizations), has an R value of 2.

        [1] Dry, Matthew; Preiss, Kym; and Wagemans, Johan (2012) "Clustering, Randomness, and Regularity: Spatial Distributions and Human Performance on the Traveling Salesperson Problem and Minimum Spanning Tree Problem," The Journal of Problem Solving: Vol. 4 : Iss. 1, Article 2. 
        [2] G. C. Crişan, E. Nechita and D. Simian, "On Randomness and Structure in Euclidean TSP Instances: A Study With Heuristic Methods," in IEEE Access, vol. 9, pp. 5312-5331, 2021, doi: 10.1109/ACCESS.2020.3048774.
        [3] Clark, Philip J., and Francis C. Evans. “Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations.” Ecology, vol. 35, no. 4, 1954, pp. 445–53. JSTOR, https://doi.org/10.2307/1931034. Accessed 12 Dec. 2022.

        Returns:
            float: R index
        """
        vert = self.instance.node_coords
        vert_arr = np.array([(x[0], x[1]) for x in vert.values()])

        hull = ConvexHull(vert_arr)
        dm = self.distance_matrix.copy()

        np.fill_diagonal(dm, np.inf)
        nearest_neighbor_sum = np.sum(np.amin(dm, axis=1))

        R = np.sqrt(2)*nearest_neighbor_sum / \
            np.sqrt(hull.volume*self.dimension)
        return R

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
            solution (list[int], optional): List of node indices that construct the solution path
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
            plt.close()

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
            "statistics": {
                'mean': self.get_mean_distance(),
                'median': self.get_median_distance(),
                'coeff_var': self.get_coefficient_of_variation(),
                'qdc': self.get_quartile_dispersion_coefficient(),
                'R': self.get_regularity_index(),
                'eigen1': self.get_first_eigenvalue()
            },
            "dynamic_props": dynamic_props
        }

    def __str__(self) -> str:
        dynamic_props = ""
        if self.dynamic:
            dynamic_props = "Dynamic Frequency: %s, Dynamic Intensity: %d, Minimum Iterations before Dynamic: %s\n" % (
                self.dynamic_frequency, self.dynamic_intensity, self.min_iteration_count)
        return "\nType: Symmetric TSP, Name: %s, Dimension: %s, Weight Type: %s\n" % (self.instance.name, self.dimension, self.instance.edge_weight_type) + dynamic_props

    __repr__ = __str__
