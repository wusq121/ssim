import numpy as np
from options import Options
from sympy import Matrix

class Sequences():
    def __init__(self, option: Options) -> None:
        self.bits_per_seq = option.bits_per_seq
        self.num = 2 ** self.bits_per_seq
        self.length = 2 * self.num
        self.seed = option.seed
        self.P = self._generate()
    

    def _generate(self):
        np.random.seed(self.seed)
        tmp = np.sign(np.random.randn(self.length))
        np.random.seed(None)

        # 循环移位
        mat = np.zeros([self.length, self.length])
        for i in range(self.length):
            mat[i, :] = np.roll(tmp, i)
        
        #满秩分解
        _, j = Matrix(mat).rref()
        F = mat[:, j].T

        # 标准正交化
        P = np.zeros_like(F)
        for i in range(len(j)):
            if i == 0:
                P[i, :] = F[i, :]
                P[i, :] = P[i, :] / np.linalg.norm(P[i, :])
            else:
                temp = np.zeros(self.length)
                for j in range(i):
                    temp += np.dot(F[i, :], P[j, :]) * P[j, :]
                P[i, :] = F[i, :] - temp
                P[i, :] = P[i, :] / np.linalg.norm(P[i,:])
        
        return P[:self.num, :]
