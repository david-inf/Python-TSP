# -*- coding: utf-8 -*-
"""

Solve TSP problem and diagnose solution

"""

# print(f'Invoking __init__.py for {__name__}')

from tsp_solvers.tsp import tsp_fun
from tsp_solvers.solvers_utils import rand_init_guess

from tsp_solvers.exact.brute_force import solve_brute_force
from tsp_solvers.heuristics.local_search import solve_swap
from tsp_solvers.heuristics.multi_start import solve_multi_start

from tsp_solvers.optimize import solve_tsp, solvers_list, multi_start_ls
