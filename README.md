# Tryna solve the Travelling Salesman Problem

Statistical physics and complex systems final project

## Topic description

The problem: Travelling Salesman Problem, minimum Hamiltonian cycle problem. Find the shortest route that connects all cities.

## Recipes

Perturbation methods:
- [x] `swap`: draw two random indices $\{i,j\}$, swap the corresponding cities in the sequence (canonical ensemble)
- [x] `reverse`: draw two random indices $\{i,j\}$ s.t. $i < j$, reverse the sequence between those two cities (canonical ensemble)
- [ ] `remove`: 
- [ ] `insert`: 

Algorithms:
- **Exact**:
    - Brute-force: check all possible permutations
- **Heuristics**:
    - 2-exchange: `swap` and `reverse` neighborhood operators
- **Meta-heuristics**:
    - Multi-start
    - Simulated annealing
        - `reverse` perturbation method
        - Geometric cooling schedule $T_{k+1}=\alpha T_k$

## Examples

Local search with `reverse` perturbation

https://github.com/david-inf/Python-TSP/assets/76067448/868a057a-9af2-41a6-b0de-d6feefddb97a

Simulated Annealing

https://github.com/david-inf/Python-TSP/assets/76067448/56218497-0209-496a-bf69-de4ed1858b12
