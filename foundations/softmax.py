import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_max = np.max(z)
        z = z - z_max
        exp_z = np.exp(z)
        sum_exp = np.sum(exp_z)
        result = exp_z / sum_exp
        return np.round(result,4)
