import numpy as np
from options import Options
from sympy import Matrix
from scipy.fftpack import idct, dct
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import os
import fnmatch


def file_filter(path, ext):
    flist = os.listdir(path)
    all_file = []
    for file in flist:
        filepath = os.path.join(path, file)
        if os.path.isdir(file):
            all_file.extend(file_filter(filepath, ext))
        elif fnmatch.fnmatch(filepath, '*.'+ext):
            all_file.append(filepath)
        else:
            pass
    
    return all_file


def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


class Sequences:
    """序列生成器"""
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


class Sequence_xiang2018:
    """xiang2018中的序列生成器"""
    def __init__(self, option: Options) -> None:
        self.bits_per_seq = option.bits_per_seq
        self.num = 2 ** self.bits_per_seq
        self.length = 2* self.num
        self.seed = option.seed
        self.P = self._generate()
    
    def _generate(self):
        np.random.randn(self.seed)
        tmp = np.sign(np.random.randn(self.length))
        np.random.seed(None)

        F = np.eye(self.length, dtype=np.float_)
        F[0, :] = tmp

        P = np.zeros_like(F)
        for i in range(len(self.length)):
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

class Sequences_xiang2015:
    """xiang2015中的序列生成器"""
    def __init__(self, option: Options) -> None:
        self.bits_per_seq = option.bits_per_seq
        self.num = 2 ** self.bits_per_seq
        self.length = 2 * self.num
        self.seed = option.seed
        self.P = self._generate()

    def _generate(self):
        np.random.randn(self.seed)
        tmp = np.sign(np.random.randn(self.length))
        np.random.seed(None)

        mat = np.zeros([self.length, self.length])
        for i in range(self.length):
            mat[i, :] = np.roll(tmp, i)
        
        return mat[:self.num, :]

class Audio_Segment:
    """
    音频分段器方法，受全局设置影响
    """
    def __init__(self, option: Options) -> None:
        self.num_of_frag = option.watermark_length // option.bits_per_seq
        self.frag_length = 2 ** (option.bits_per_seq + 1)
        self.frag_of_per_seg, self.num_of_segment = crack(self.num_of_frag)
        self.seg_length = self.frag_of_per_seg * self.frag_length * 8
        self.clip_length = self.seg_length * self.num_of_segment
    
    def __call__(self, audio_mono: np.ndarray, sr, feature_point):
        idx = feature_point + sr//10
        clip = audio_mono[idx: idx+self.clip_length]
        clip_mat = clip.reshape((self.num_of_segment, self.seg_length))
        clip_mat_dct = dct(clip_mat, type=2, norm='ortho')

        # 分为低频高频
        l_idx = int(self.seg_length // 8)
        clip_mat_dct_l = clip_mat_dct[:, 0:l_idx]
        clip_mat_dct_m = clip_mat_dct[:, l_idx: 2*l_idx]
        clip_mat_dct_h = clip_mat_dct[:, 2*l_idx: ]

        clip_mat_dct_m_mat = clip_mat_dct_m.reshape((self.num_of_segment, self.frag_of_per_seg, self.frag_length))

        return clip_mat_dct_m_mat, clip_mat_dct_l, clip_mat_dct_h, audio_mono


class Audio_Reconstruct:
    """
    音频分段重构
    """
    def __init__(self) -> None:
        pass

    def __call__(self, audio_mono: np.ndarray, sr, feature_point, clip_mat_dct_m_mat, clip_mat_dct_l, clip_mat_dct_h):
        idx = feature_point + sr//10
        num_of_segment, frag_of_per_seg, frag_length = clip_mat_dct_m_mat.shape
        clip_mat_dct_m = clip_mat_dct_m_mat.reshape((num_of_segment, frag_of_per_seg * frag_length))
        clip_mat_dct = np.concatenate((clip_mat_dct_l, clip_mat_dct_m, clip_mat_dct_h), axis=0)
        clip_mat = idct(clip_mat_dct, type=2, norm='ortho')
        clip = clip_mat.reshape(-1)


        audio_mono[idx: idx + len(clip)]
        return audio_mono


class Feature_Dete:
    """
    音频特征点提取
    """
    def __init__(self, option: Options) -> None:
        self.distance = int(2 * option.watermark_length * (2 ** (option.bits_per_seq + 4)) / option.bits_per_seq)
        self.sigma = option.sigma

    def __call__(self, audio_mono):
        y = gaussian_filter(audio_mono, sigma=self.sigma, mode='constant')
        d = np.zeros_like(y)
        d[0] = y[0]
        for i in range(1, d.size):
            d[i] = y[i] - y[i - 1]
        idx, _ = find_peaks(np.abs(d), distance=self.distance)
        return idx


class Watermark:
    def __init__(self, watermark, option: Options) -> None:
        self.bits_per_seq = option.bits_per_seq
        self.watermark_length = option.watermark_length
        if type(watermark) is str:
            self.watermark_str = watermark
            self.watermark_np = self._str2np()
        else:
            self.watermark_np = watermark
            self.watermark_str = self._np2str()
    

    def _str2np(self):
        wmbits = []
        for i in range(0, self.watermark_length, self.bits_per_seq):
            if (i + self.bits_per_seq) > self.watermark_length:
                wmbits.append(int(self.watermark_str[i:], base=2))
            else:
                wmbits.append(int(self.watermark_str[i: (i + self.bits_per_seq)], base=2))
            
            return np.array(wmbits, dtype=np.int_)
    
    def _np2str(self):
        wm = ''
        for i in self.watermark_np:
            wm += format(i, '0b').zfill(self.bits_per_seq)
        return wm
