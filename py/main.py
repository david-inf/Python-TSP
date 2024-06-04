# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from tsp import generate_cities, circle_cities, create_city

# %% first try

D1, C1 = generate_cities(5)
cities1 = create_city(C1)

plt.scatter(C1[:,0], C1[:,1])
plt.show()

# %% second try

D2, C2, C2pol = circle_cities(50)

plt.scatter(C2[:,0], C2[:,1])
plt.xlim((-6,6))
plt.ylim((-6,6))
plt.show()

plt.polar(C2pol[:,0], C2pol[:,1])
plt.show()
