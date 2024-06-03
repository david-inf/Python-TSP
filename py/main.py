# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from tsp import generate_cities

# %% first try

mat1 = generate_cities(5)

plt.scatter(mat1[:,0], mat1[:,1])
plt.show()


