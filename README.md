# Tryna solve the Travelling Salesman Problem

Statistical physics and complex systems final project

## Topic description

The problem: Travelling Salesman Problem, minimum Hamiltonian cycle problem. Find the shortest route that connects all cities.

## Recipes
Permutation encoding `[0,3,4,1,2,0]`

Perturbation methods:
- [x] `swap`: draw two random indices $\{i,j\}$, swap the corresponding cities in the sequence (canonical ensemble)
- [x] `reverse`: draw two random indices $\{i,j\}$ s.t. $i < j$, reverse the sequence between those two cities (canonical ensemble)
- [ ] `remove`: 
- [ ] `insert`: 

Algorithms:
- **Exact**:
    - Brute-force: check all possible permutations and select the best one
- **Greedy**
    - Nearest Neighbor with variants:
        - `exact`: the next node is the nearest one (lower cost)
        - `random`: the next node is randomly chosen
        - `weighted`: the next node is randomly chosen cost-weighted
- **Heuristics**:
    - 2-exchange variants:
        - `swap`: draw two nodes and swap their position
        - `reverse`: draw two nodes and reverse the sequence inclusive
- **Meta-heuristics**:
    - Multi-start: a pool of random initial solutions for 2-exchange and SA
    - Simulated Annealing (SA)
        - `reverse` perturbation method
        - Geometric cooling schedule $T_{k+1}=\alpha T_k$

## Examples

Local search with `reverse` perturbation

https://github.com/david-inf/Python-TSP/assets/76067448/868a057a-9af2-41a6-b0de-d6feefddb97a

Simulated Annealing

https://github.com/david-inf/Python-TSP/assets/76067448/56218497-0209-496a-bf69-de4ed1858b12
