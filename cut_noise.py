# -*- coding: utf-8 -*-
"""
裁剪掉收集的recaptcha v2音频数据集两边的多余的噪声
这个方法并不是完好的方法，弄完之后还得一一核对是否裁剪合理，不合理的需要手动修改保存时的参数
注：前提是所要处理的音频文件已经是 int16、16000Hz、单声道、wav格式文件。
"""
import sys
import os
from scipy.io.wavfile import read, write
from glob import glob
import numpy as np
import pandas as pd
import shutil
from ffmpy3 import FFmpeg
import matplotlib.pyplot as plt
import random


def recaptcha_v2_process(wav_dirs, save_dirs):
    """
    裁剪掉收集的recaptcha v2音频数据集两边的多余的噪声
    这个方法并不是完好的方法，弄完之后还得一一核对是否裁剪合理，不合理的需要手动修改保存时的参数
    """
    wav_list = glob(os.path.join(wav_dirs, "*"))

    for i in range(0, len(wav_list)):
        name = os.path.basename((wav_list[i]))
        print(name)
        sr, audio = read(wav_list[i])
        audios = np.abs(np.int64(audio))
        # audios = np.power(audios, 1)
        r = 10
        s = []
        for i1 in range(0, len(audios), r):
            si = 20*np.log10(np.sum(np.square(audios[i1: i1+r])))
            s.append(si)
           # if si < 145 and i1 <= len(audios)/2:
            if si < 143 and i1 <= len(audios)/2:
                audios[i1: i1+r] = 0
            if si < 144 and i1 > len(audios)/2:
                audios[i1: i1+r] = 0

        r1 = np.ceil(np.array(s)/10)
        d = {}
        for i2 in r1:
            d[i2] = d.get(i2, 0) + 1

        l = np.nonzero(audios)[0]
        l1, l2 = l[0], l[-1]
        print(l1, " ", l2)
        # write(os.path.join(save_dirs, name), sr, audio[l1-100:l2+3300])  # 大多数情况下-500没什么问题，但是后面+4000这个有问题
        """by xiaoyan"""
        # 这里稍稍有点问题，观察发现，后半部分的噪音没有裁剪掉，导致音频长度大于65300，执行main的时候，由于超过最大长度65536，会报错
        # 对于去噪声后长度大于65300的部分进行裁剪，裁剪后半段的噪声
        if(len(audio[l1-100:l2+3300])>50000):
            write(os.path.join(save_dirs, name), sr, audio[l1-100:l1-100+50000])
        else:
            write(os.path.join(save_dirs, name), sr, audio[l1-100:l2+3300])

        # break


if __name__ == "__main__":
    wav_dirs = r"/media/ps/data/YZN/Adv_audio/data/recaptchaV2/recaptcha1wnew_transf"  # 原文件夹，注意这里面是单声道、int16、16000Hz的wav文件
    save_dirs = r"/media/ps/data/YZN/Adv_audio/data/recaptchaV2/recaptcha1wnew_crop" # 保存的文件夹
    recaptcha_v2_process(wav_dirs, save_dirs)
