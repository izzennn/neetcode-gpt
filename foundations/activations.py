import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        h = np.zeros(len(z))
        for i in range(len(z)):
            h[i] = np.round(1/(1+np.exp(-z[i])), 5)
        return h

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        h = np.zeros(len(z))
        for i in range(len(z)):
            h[i] = max(0,z[i])
        return h
