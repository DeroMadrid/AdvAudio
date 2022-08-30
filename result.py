"""
对已经生成的音频对抗样本进行计算，统计其转录效果，包括转录正确率、转录的stoi、音频信噪比。
"""
from scipy.io.wavfile import read, write
import numpy as np
import os
from glob import glob
from metric import snr, psnr, segsnr, stoi_audio, get_pesq, tran_acc


if __name__ == "__main__":
    prefix = "/media/ps/data/YZN/Adv_audio/result/recaptcha"
    times = "03-01-08-23"   # 修改
    org_audios = glob(os.path.join(prefix, times, "validation", "org_audio"))
    adv_audios = glob(os.path.join(prefix, times, "validation", "adv_audio"))
    tran_txt = os.path.join(prefix, times, "validation_tran.txt")
