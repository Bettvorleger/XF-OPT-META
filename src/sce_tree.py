import math
import copy
from treelib import Tree
from problem import Problem


class SCENode:

    def __init__(self, pers_best: tuple, pb_quality: float, pers_prev: tuple, pp_quality: float):
        """
        Creates an SCE node object for the hierarchical tree structure

        Args:
            pers_best (tuple): Personal best solution of the SCE node
            pb_quality (float): Quality of the personal best solution
            pers_prev (tuple): Previous solution of the SCE node
            pp_quality (float): Quality of the personal pervious solution
        """
        self.pers_best = pers_best
        self.pb_quality = pb_quality
        self.pers_prev = pers_prev
        self.pp_quality = pp_quality


class SCETree:

    def __init__(self, init_solution: Problem.init_solution, node_count=13, child_count=3):
        """
        Creates the hierarchical tree for SCE solution storage and dynamic detection

        Args:
            init_solution (Problem.init_solution): Method for randomly generating new solutions 
            node_count (int): Number of nodes to be placed inside the tree. Defaults to 13.
            child_count (int): Max number of childs for each node (k-ary tree). Defaults to 3, so a ternary tree
        """
        self.nodes = node_count
        self.childs = child_count
        self.tree = Tree()
        self.solution_generator = init_solution

    def init_tree(self):
        """
        Initialize the SCE tree a k-ary tree (depens on chƒilds value)
        and give each SCE node initial random values.
        """

        for i in range(0, self.nodes):
            if self.tree.root is None:
                p = None  # root node
            else:
                p = math.floor((i-1)/self.childs)

            pers_best, pb_quality = self.solution_generator()
            pers_prev, pp_quality = self.solution_generator()

            self.tree.create_node(
                "SCE_"+str(i), i, parent=p, data=SCENode(pers_best, pb_quality, pers_prev, pp_quality))

        # self.tree.show()

    def reset(self):
        """
        Reset the SCE tree
        """
        self.tree = Tree()

    def update_node(self, nid: int, solution: tuple, solution_quality: float) -> None:
        """
        Update the solution data of a node given a new solution and its quality

        Args:
            nid (int): Index of a node within the tree
            solution (tuple): List of a possible solution to the problem
            solution_quality (float): Quality of the provided solution
        """
        nd = self.tree.get_node(nid).data

        # update personal previous
        nd.pers_prev, nd.pp_quality = solution, solution_quality

        # update personal best
        if solution_quality < nd.pb_quality:
            nd.pers_best, nd.pb_quality = solution, solution_quality

    def set_node_quality(self, nid: int, pp_quality: float, pb_quality: float) -> None:
        """
        Set the values for personal previous and personal best solution quality.

        Args:
            nid (int): Index of a node within the tree
            pp_quality (float): Quality of the personal previous solution
            pb_quality (float): Quality of the personal best solution
        """
        nd = self.tree.get_node(nid).data
        nd.pp_quality, nd.pb_quality = pp_quality, pb_quality

    def swap_nodes(self, child: int, parent: int) -> None:
        """
        Swap the places of the child and parent node in the tree, moving also the data.
        All the parent (predecessor) and child (successors) references have to be changed,
        following the logic:
            ref(child) -> parent
            ref(parent) -> child

        Args:
            child (int): Index of the child node
            parent (int): Index of the parent node

        Raises:
            IndexError: Node in first argument has to be a child of the parent node in the second argument
        """

        # Getting a deep copy of the childs succesors list for assignment to parent node
        cn = self.tree.get_node(child)
        cn_successors = copy.deepcopy(cn._successors)
        pn = self.tree.get_node(parent)

        if cn._predecessor[self.tree._identifier] is not parent:
            raise IndexError(
                "Node in first argument is no child of parent node in the second argument")

        # Change all the parents of the former child node to the new parent
        for c in self.tree.children(child):
            c._predecessor[self.tree._identifier] = parent

        # Change all the parents of the former parent node to the new parent
        for pc in self.tree.children(parent):
            pc._predecessor[self.tree._identifier] = child

        # Change childs of the grandparent (parent of parent) to former child
        # or assign former child as tree root, if parent was root
        if self.tree.level(parent) > 0:
            gp = self.tree.parent(parent)
            gp._successors.get(self.tree._identifier).remove(parent)
            gp._successors.get(self.tree._identifier).append(child)
        else:
            self.tree.root = child

        # Swap all the successors and predecessor of child and parent node directly
        cn._successors = pn._successors
        cn._successors.get(self.tree._identifier).remove(child)
        cn._successors.get(self.tree._identifier).append(parent)
        cn._predecessor[self.tree._identifier] = gp._identifier if self.tree.level(
            parent) > 0 else None

        pn._successors = cn_successors
        pn._predecessor[self.tree._identifier] = child

    def change_handling(self, reaction_type: str) -> None:
        """
        Trigger the change handling procedure for resetting certain parts of the tree,
        upon detection of dynamic changes to the problem instance.
        Implemented according to the paper, H_full and H_partial, resetting only the personal best values.

        Args:
            reaction_type (str): Type of reaction/resett, H_full is 'full' and H_partial is 'partial'
        """
        if reaction_type == 'full':
            # resetting the whole tree
            self.reset_pers_best(level=0)
        elif reaction_type == 'partial':
            # resetting only from the second level down
            self.reset_pers_best(level=2)

    def reset_pers_best(self, level=0) -> None:
        """        
        Reset the nodes pers_best and pb_quality data within the SCE tree, including the node.
        If a level is provided, the reset is starting from that level (e.g. level=1 resets the whole tree, except the root).
        New values for the reset data will be initialized.

        Args:
            level (int, optional): Level from which to start resetting. Defaults to 0 (root).
        """
        nodes = list(self.tree.filter_nodes(
            lambda x: self.tree.depth(x) >= level))
        for n in nodes:
            n.data.pers_best, n.data.pb_quality = self.solution_generator()

    def get_solution(self, nid: int) -> tuple[list, int]:
        """
        Get the personal (previous) solution of a node 

        Args:
            nid (int): Node index within the SCE tree

        Returns:
            tuple(list, int): Tuple of the solution path and the corresponding solution quality (e.g. tour length)
        """
        return self.tree.get_node(nid).data.pers_prev, self.tree.get_node(nid).data.pp_quality

    def get_best_solution(self, nid: int) -> tuple[list, int]:
        """
        Get the personal best solution of a node

        Args:
            nid (int): Node index within the SCE tree

        Returns:
            tuple(list, int): Tuple of the solution path and the corresponding solution quality (e.g. tour length)
        """
        return self.tree.get_node(nid).data.pers_best, self.tree.get_node(nid).data.pb_quality

    def get_parent_best_solution(self, nid: int) -> list:
        if self.tree.root == nid:
            return self.tree.get_node(nid).data.pers_best
        else:
            pid = self.tree.parent(nid).identifier
            return self.tree.get_node(pid).data.pers_best

    def get_populations(self, nid: int) -> tuple[dict[int]]:
        """
        Returns the population of the SCE (meaning the personal previous, the personal best and the parent best solution path) as tuple.
        The solutions themselves are parsed as dictionaries, with the nodes as key and index in the list as value, so each entry is (node: index) 
        Reason: Dicts  can be searched in O(1), compared to lists/tuples in O(n)

        Args:
            nid (int): Node index within the SCE tree

        Returns:
            tuple[dict[int]]: Tuple of populations with solutions in dicts as (node: index)
        """
        if self.tree.root == nid:
            return {k: v for v, k in enumerate(self.tree.get_node(nid).data.pers_prev)}, {k: v for v, k in enumerate(self.tree.get_node(nid).data.pers_best)}, {k: v for v, k in enumerate(self.tree.get_node(nid).data.pers_best)}
        else:
            pid = self.tree.parent(nid).identifier
            return {k: v for v, k in enumerate(self.tree.get_node(nid).data.pers_prev)}, {k: v for v, k in enumerate(self.tree.get_node(nid).data.pers_best)}, {k: v for v, k in enumerate(self.tree.get_node(pid).data.pers_best)}

    def get_parent(self, nid: int) -> int:
        """
        Get the parent for a given node index. 
        If the id is root, return the id itself (root is its own parent).

        Args:
            nid (int): Node index within the SCE tree

        Returns:
            int: Index of the parent node
        """
        if self.tree.root == nid:
            return nid
        else:
            return self.tree.parent(nid).identifier

    @property
    def all_nodes_top_down(self) -> list:
        """
        Return all node indices in a breadth-first (here top-down) order,
        starting with the root node

        Returns:
            list: A top-down list of node indices.
        """
        return list(self.tree.expand_tree(mode=Tree.WIDTH))

    def get_info(self) -> dict:
        """
        Get information about the current SCE tree as a dict

        Returns:
            dict: Information about the SCE tree
        """

        return {
            "num_sce_nodes": self.nodes,
            "child_per_node": self.childs
        }

    def __str__(self) -> str:
        return "\nSCE Nodes: %s, Childs per Node: %s\n" % (self.nodes, self.childs)

    __repr__ = __str__
