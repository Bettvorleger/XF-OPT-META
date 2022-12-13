# XF-OPT/META
## E**X**perimentation **F**ramework and (Hyper-)Parameter **Opt**imization for **Meta**heuristics

This python package provides and easy-to-use and modular framework for implementing metaheuristic algorithms and corresponding problems (e.g., TSP, QAP). Futhermore it provides functions/modes for optimizing the paramters used in the metaheuristics using Hyperparamter optimization algorithms (e.g., Random Search), analyzing multiple runs of an implemented metaheurisitc algorithm or simply running said algorithm interactively.

### 1. Overview

As of now, the package provides a CLI-based control over its features.
Following the installtion instructions, one should be able to start the programm using:
```
python main.py
```
Please refer to the help section for information on the optional parameters:
```
python main.py -h
```

There are three pre-definded use cases:

- Run Mode (--mode run, default)
- Experimentation Mode (--mode exp)
- Optimization Mode (--mode opt)

Each case is provided with a default TSPLIB problem instance (rat195), that can be changed as well, even using different problem types, like QAP for example.

That being said, the package constists of multiple python modules/classes, that can be used interchangeably and connected as neccesary without the use of the CLI.

A web-based user interface and dashboard for graphs is also planned.

### 2. Installation

***Note:**\
This package only works with Python version >= 3.10.
But, since speed is a major factor, when it comes to metaheuristics, Python version 3.11 is recommended due to even further performance improvements.*

Start by cloning the repository to your local system.
Then, open a terminal in the `/src` directory, where the python code is located, and install the dependencies, either directly via pip
```
pip install -r requirements.txt
```
or using you favorite python environment manager (e.g., venv or conda).

Now you are ready to start the script using
```
python main.py
```


### 3. Experimental Setup and Tests

## 3.1. Choice of problem instances
For the problem instance category, the symmetric TSP was chosen, not only because it relates to previous work done with this algorithm (see paper), but also because it can be generalized to lots of different, relevant problems, especially in logistics. 

The TSP instance test cases will be of the popular TSPLIB, since they are tried and tested by lots of publications and also have the benefit of having the optimal solution known for each of the problems.
Besides giving the weights in standard 2D Euclidean form, the are also geographical distance problem instances or distance matrices. The thesis will focus on Eucledian 2D instances for simplicity, but the software itself can handle any type of TSP edge weight type.

The thesis will focus especially on the dynamic TSP (as explained in the chapter before). However, the dynamic part is implemented by the package itself, not the TSPLIB.
All TSPLIB instances have varying number of cities (henceforth called dimension) and oftentimes certain characteristics by which the cities are placed in the space, sometimes describes within the TSPLIB file (under COMMENT).
To quantify these instances the mean and median distance between nodes was also calculated. This gives the opportunity to select a meaningful, disjunct subset of problem instances, without using too many, since the computational effort to run the larger instances is pretty significant (more on that later).
The selection of problem instances is influenced by two metrics: dimension and city placement characteristic.
Since this implementation of the HSPPBO algorithm scales linearly with the dimension n, so O(n) time complexity, and the process of hyperparameter optimization runs the algorithm around hundred times, with the optimization being repeated several times for each dynamic configuration and problem instance, the maximum dimension used will be around 400. The lower bound for the dimension n will be 50, since smaller instances, will make it difficult to make out any placement characteristics. This results in dimension bound by the interval [50,450], which is then roughly categorized into small instances (50-150 cities), medium instances (150-300 instances) and large instances (300-450) instances.
The city placement characteristic is determined with the help of the following statistical values calculated for each TSPLIB instance using the corresponding distance matrix:

- mean
- median
- coeff_var: coefficient of variation (https://en.wikipedia.org/wiki/Coefficient_of_variation)
- qdc: quartile coefficient of dispersion (https://en.wikipedia.org/wiki/Quartile_coefficient_of_dispersion)
- R: regularity index (according to [1] and [2])
- eigen1: the first eigenvalue (according to [3])
- eigen_gap

Three methods for determining the placement characteristic are possible:
1. Using value ranges for the regularity index R to distinct between structures (according to [1] and [2])
2. Using the value ranges of the gap between the first two (largest) eigenvalues to distinct between structures (according to [4])
3. Using KMeans clustering with the parameters qdc, R and eigen1 to automatically generate clusters.

Since methods 1 and 2 rely on value ranges, they already give the inherent structural property of each value range. These are:
- heavy_cluster
- semi_cluster
- random
- semi_regular
- regular

For method 3, however, due to the nature of KMeans clustering, there are no resulting structural properties suggested for the clusters. One could only imply the above mentioned properties by looking at the ranges and visualizations for the instances being clustered together.

Researching and applying each of these methods resulted in a mixed outcome. 
The first method worked generally well, being able to categorize each problem into a satisfying group. But, it was heavily influenced by artificial patterns, such as pcb442 or ts225, classifying these as random.
The second method proved unpractical to implement, since it could not be directly calculated on the distance matrix and instead would use its Laplacian matrix. The paper [4] also bases its theory upon a positive semi-definite matrix, which the distance matrix is not.
The third method resulted in pretty consistent clusters and even managed to separate most of the artificial patterns from the rest, especially through the use of the eigenvalue. 
The 3D scatter plot for the KMeans clustering results is shown here:
![image](problems/clusters_kmeans.png)


[1] Dry, Matthew; Preiss, Kym; and Wagemans, Johan (2012) "Clustering, Randomness, and Regularity: Spatial Distributions and Human Performance on the Traveling Salesperson Problem and Minimum Spanning Tree Problem," The Journal of Problem Solving: Vol. 4 : Iss. 1, Article 2. 

[2] G. C. Crişan, E. Nechita and D. Simian, "On Randomness and Structure in Euclidean TSP Instances: A Study With Heuristic Methods," in IEEE Access, vol. 9, pp. 5312-5331, 2021, doi: 10.1109/ACCESS.2020.3048774.

[3] Cvetković, Dragoš, et al. "THE TRAVELING SALESMAN PROBLEM." Bulletin (Académie serbe des sciences et des arts. Classe des sciences mathématiques et naturelles. Sciences mathématiques) 43 (2018): 17-26.

[4] Lovász, László Miklós. “Eigenvalues of graphs.” (2007).



