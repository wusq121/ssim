from options import Options
import numpy as np
from utils import *
import os
import librosa
import soundfile as sf


class Embedder:
    def __init__(self, option:Options) -> None:
        self.P = Sequences(option).P
        self.segment = Audio_Segment(option)
        self.reconstruct = Audio_Reconstruct()
        self.fp_dete = Feature_Dete(option)
        self.option = option
        self.data_dir = option.data_dir
        self.out_dir = option.out_dir
        self.peaq_test = option.peaq_test_dir
        self.logfile = option.logfile

    def _weight(self, X):
        """
        Shape of X is n_frag * frag_length
        Shape of P is n_seq * frag_length
        """
        P = self.P
        tmp = np.abs(np.dot(X, P.T))
        max_per_X = np.max(tmp, axis=1)
        max_per_X[max_per_X < 0.02] = 0.02
        return max_per_X.reshape(-1, 1)


    def _wm_embed(self, X_m, wmbits):
        """
        在一个clip中嵌入音频
        """
        P = self.P
        n_s, n_frg, frag_length = X_m.shape
        n_seq, frag_length = P.shape
        if type(wmbits) is int:
            wmbits = wmbits * np.ones(n_frg * n_s, dtype=np.int_)
        
        wmbits = wmbits.reshape((n_s, n_frg))
        Y = np.zeros_like(X_m)
        for i in range(n_s):
            wm_t = wmbits[i, :]
            Xm_t = X_m[i, :, :]
            P_t = P[wm_t, :]
            w = self._weight(Xm_t)
            sign = np.sign(np.sum(Xm_t * P_t, axis=1)).reshape(-1, 1)
            Y[i, :, :] = Xm_t + w * sign * P_t
        return Y
    
    def outfilename(self, inputfilename):
        """
        根据输入信号的文件名给出
        """
        basename = os.path.basename(inputfilename)
        filename, ext = os.path.splitext(basename)
        outfile_em = os.path.join(self.out_dir, basename)
        peaq_ori = os.path.join(self.peaq_test, filename, "origin"+ext)
        peaq_stego = os.path.join(self.peaq_test, filename, "stego"+ext)
        return outfile_em, peaq_ori, peaq_stego
    

    def _embed_mono(self, audio, sr, wmbits):
        idx = self.fp_dete(audio)
        for i in idx:
            X_m, X_l, X_h, y = self.segment(audio, sr, i)
            Y_m = self._wm_embed(X_m, wmbits)
            y = self.reconstruct(y, sr, i, Y_m, X_l, X_h)
        return y
    
    def __call__(self, audio_path: str, watermark: Watermark):
        sr = librosa.core.get_samplerate(audio_path)
        audio, sr = librosa.core.load(audio_path, sr, mono=False)
        stego = np.zeros_like(audio)
        wmbits = watermark.watermark_np
        if audio.ndim > 1:
            for i in range(audio.shape[0]):
                y = audio[i, :].flatten()
                stego[i, :] = self._embed_mono(y, sr, wmbits)
        else:
            stego = self._embed_mono(audio, sr, wmbits)
        
        outfile_em, peaq_ori, peaq_stego = self.outfilename(audio_path)

        ori_p = librosa.core.resample(audio, sr, 48000)
        stego_p = librosa.core.resample(stego, sr, 48000)

        sf.write(outfile_em, stego.T, sr)
        sf.write(peaq_ori, ori_p.T, 48000)
        sf.write(peaq_stego, stego_p.T, 48000)