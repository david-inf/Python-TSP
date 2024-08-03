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
        - Automatic $T_0$ tuning based on acceptance rate $\chi$
        - Geometric cooling schedule $T_{k+1}=\alpha T_k$
    - Genetic Algorithm (GA)

## Examples

| **Algorithm** | **Circular layout** | **Random layout** |
| -- | -- | -- |
| Nearest Neighbor with `weighted` | ![circle-greedy2](https://github.com/user-attachments/assets/2b2410af-60ea-4b29-9604-52ba4584e94e) | ![rand-greedy2](https://github.com/user-attachments/assets/3213cb78-0bdf-4104-93d4-e7768e29f0d1) |
| Best multi-start local search with `swap` | ![circle-swap](https://github.com/user-attachments/assets/a257d403-8fb9-4505-a2f5-4e478fefc798) | ![rand-swap](https://github.com/user-attachments/assets/f5a2778c-a5bd-431b-8f55-a6dd2304f668) |
| Best multi-start local search with `reverse` | ![circle-reverse](https://github.com/user-attachments/assets/d55ad505-018b-473d-b503-d9053ed0fb03) | ![rand-reverse](https://github.com/user-attachments/assets/92cce929-bb47-4273-9e14-1e083f12ab86) |
| Best multi-start Simulated Annealing | ![circle-annealing-quad](https://github.com/user-attachments/assets/21eaaa01-8f07-4ee4-a5c0-cedc51422c3a) | ![rand-annealing-quad](https://github.com/user-attachments/assets/02a173f5-f90f-4520-a806-2f1681d89743) |
