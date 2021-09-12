from options import Options
import numpy as np
from utils import *


class Embeddor:
    def __init__(self, option:Options) -> None:
        self.P = Sequences(option).P
        self.segment = Audio_Segment(option)
        self.reconstruct = Audio_Reconstruct()
        self.fp_dete = Feature_Dete(option)
        self.option = option

    def _weight(X, P):
        """
        Shape of X is n_frag * frag_length
        Shape of P is n_seq * frag_length
        """
        tmp = np.abs(np.dot(X, P.T))
        max_per_X = np.max(tmp, axis=1)
        max_per_X[max_per_X < 0.02] = 0.02
        return max_per_X.reshape(-1, 1)


    def _wm_embed(self, X_m, P, wmbits):
        """
        在一个seg中嵌入音频
        """
        n_frg, frag_length = X_m.shape
        n_seq, frag = P.shape
        if type(wmbits) is int:
            wmbits = wmbits * np.ones(n_frg, dtype=np.int_)
        
        P_t = P[wmbits, :]
        w = self._weight(X_m, P)
        sign = np.sign(np.sum(X_m * P_t, axis=1)).reshape(-1, 1)
        Y = X_m + w * sign * P_t
        return Y
    
    
