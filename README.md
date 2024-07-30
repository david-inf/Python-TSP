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
    - Genetic Algorithm (GA)

## Examples

Local search with `reverse` perturbation

https://github.com/user-attachments/assets/7becbec8-25e1-456b-a1bf-647b750770f2

https://github.com/user-attachments/assets/f87c32c4-de39-4a50-8d16-7f8578f9bd6f

Simulated Annealing

https://github.com/user-attachments/assets/94effa22-3148-4c44-a937-31941d687b15

https://github.com/user-attachments/assets/9fc79aff-aa3a-4e3f-9e7d-08c52f572358
