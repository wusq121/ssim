from options import Options
import numpy as np
from utils import *
import os
import librosa
import soundfile as sf

class Extractor:
    def __init__(self, option: Options) -> None:
        self.P = Sequences(option).P
        self.segment = Audio_Segment(option)
        self.fp_dete = Feature_Dete(option)
        self.option = option
        self.data_dir = option.out_dir
        self.logfile = option.logfile


    def _wm_extract(self, Y_m):

        """
        变量: n_s, Y中的segment的数量
        变量: n_frag, segment中的fragment数量
        变量: n_frag, 每个fragmet的长度，是Y和P共有的属性
        """
        P = self.P
        n_s, n_frag, frag_length = Y_m.shape
        n_seq, frag_length = P.shape

        wmbits = np.zeros((n_s, n_frag))
        for i in range(n_s):
            for j in range(n_frag):
                tmp = np.abs(np.dot(Y_m[i, j, :], P.T))
                if tmp.all() == 0:
                    return -1
                else: 
                    wmbits[i, j] = tmp.argmax()
        return wmbits
    

    def _vote(self, wmbits):
        wm = np.zeros_like(wmbits[0, :, :])
        for i in range(wm.shape[0]):
            for j in range(wm.shape[1]):
                arr = wmbits[:, i, j]
                tu = sorted([(np.sum(arr==k), k) for k in set(arr)])
                wm[i, j] = tu[-1][1]
        return wm
    
    def extract_mono(self, audio, sr):