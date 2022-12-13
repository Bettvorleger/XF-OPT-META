from abc import ABC, abstractmethod


class Problem(ABC):
    TYPE = None
    """
    Abstract method (interface) for possible problem instances.
    """

    @abstractmethod
    def set_dynamic(self):
        """
        Set the dynamic parameters of the problem, if it's to be a dynamic problem
        """
        pass

    @abstractmethod
    def set_random_seed(self):
        """
        Fixing the seed of the RNG, making the results predictable
        """
        pass

    @abstractmethod
    def check_dynamic_change(self):
        """
        Checks if dynamic change should happen in current iteration and triggers it if necessary
        """
        pass

    @abstractmethod
    def init_solution(self):
        """
        Initialize an new solution for the problem.
        TSP: Create list of random indices with length the same as number of cities
        """
        pass

    @abstractmethod
    def get_solution_quality(self):
        """
        Get the solution quality for the given possible solution.
        TSP: Length of the given tour 
        """
        pass
    
    @abstractmethod
    def get_heuristic_component(self):
        """
        Get the heurstic component of the problem, in general or for specified indices
        """
        pass

    @abstractmethod
    def get_optimal_solution(self):
        """
        Get the optimal solution (quality) of the problem, if existent 
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Create an interactive view or image from the problem instance
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """
        Get information about the current problem instance as a dict

        Returns:
            dict: Information about the TSP instance
        """
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass

    @dimension.setter
    def dimension(self, value):
        self.dimension = value
        


    

   
