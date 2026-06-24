import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        preds = np.dot(x,w) + b
        preds = 1/(1+np.exp(-preds))
        loss = np.mean(0.5 * (preds-y_true)**2)
        dw = (preds - y_true) * preds*(1-preds)*x
        db = (preds - y_true) * preds*(1-preds)

        return (np.round(dw,5), np.round(db,5))
