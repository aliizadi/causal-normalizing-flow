import numpy as np
from functools import reduce


D = 5
masks = {}
hidden_dims = [5, 5, 5]
L = len(hidden_dims)
masks[0] = np.arange(D)
for l in range(L):
     low = masks[l].min()
     size = hidden_dims[l]
     # masks[l+1] = np.random.randint(low=low, high=D-1, size=size)
     masks[l+1] = np.array([i for i in range(5)]) 

masks[L+1] = masks[0]

masks_mat = [(masks[l][:, None] <= masks[l+1][None, :]).T.astype(int) for l in range(L)]
masks_mat.append((masks[L][:, None] < masks[L+1][None, :]).T.astype(int))
reduce(np.dot, reversed(masks_mat))

