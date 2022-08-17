import numpy as np
import numpy_indexed as npi

allNodes = np.array([
             [1,1,1],
             [1,2,1],
             [1,4,5],
             [1,2,1],
             [1,1,1]])

nodes, index = np.unique(allNodes, return_index=1, axis=0)

print(index)
print(nodes)

