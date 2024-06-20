# Travelling salesman problem

Statistical physics and complex systems final project

## Topic description

The problem: Travelling Salesman Problem, minimum Hamiltonian cycle problem. Find the shortest route that connects all cities.

Given a set of cities:
**place image**

Find the shortest possible route that connects all cities:
**place image with feasible solution**

Perturbation methods:
- `swap`: draw two random indices $\{i,j\}$, swap the corresponding cities in the sequence (canonical ensemble)
- `reverse`: draw two random indices $\{i,j\}$ s.t. $i < j$, reverse the sequence between those two cities (canonical ensemble)
- `insert`: 


Algorithms:
- Exact:
    - Brute-force: check all possible permutations
- Heuristics:
    - 2-exchange: `swap` and `reverse` neighborhood operators
    - k-exchange??
- Meta-heuristics
    - Multi-start
    - Simulated annealing
        - `reverse` perturbation method
        - Geometric cooling schedule $T_{k+1}=\alpha T_k$

## Examples

Local search with `reverse` perturbation

<video src="./py/anims/local_search-rev.mp4" width="320" height="240" controls></video>

Simulated Annealing

<video src="./py/anims/sim-annealing.mp4" width="320" height="240" controls></video>
