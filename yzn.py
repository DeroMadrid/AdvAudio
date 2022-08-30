# coding=utf-8
from scipy.io.wavfile import read, write
import numpy as np
import tensorflow as tf
import sys
from glob import glob
import librosa
from share import toks
import os
from share import toks, ds_preds, ds_preds_lm, ds_var
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_p = "/media/ps/data/YZN/Adv_audio/result/recaptcha/03-24-21-56_wgan-gptarget_libri/validation/adv_audio"
train_p = "/media/ps/data/YZN/Adv_audio/result/recaptcha/03-24-21-56_wgan-gptarget_libri/validation/org_audio"
# wavs = glob(os.path.join(train_p, "*"))
# for i in range(len(wavs)):
#     sr, data = read(wavs[i])
#     valid_len = np.where(abs(data) > 0)[0][-1] + 1
#     savename = wavs[i]
#     write(savename, sr, data[:valid_len+1])


test_wavs = glob(os.path.join(test_p, "*"))
train_wavs = glob(os.path.join(train_p, "*"))
print(len(test_wavs))
print(test_wavs[-1])
# for w in train_wavs:
#     name = os.path.basename(w).split("_")[0]
#     ids = int(name)
#     if ids < 82000:
#         os.remove(w)

    # if os.path.join(test_p, name) in test_wavs:
    #     print(w)
    #     os.remove(w)














# temp = []
# d = {}
# for i in range(len(wavs)):
#     name = os.path.basename(wavs[i]).split("-")[0]
#     temp.append(name)
#     if name in d:
#         d[name] += 1
#     else:
#         d[name] = 1
#
# temp = list(set(temp))
# print(len(temp))
#
# d = sorted(d.items(), key=lambda x: x[1], reverse=True)
# for i in range(len(d)):
#     if d[i][1] < 8:
#         temp.remove(d[i][0])
# print(len(temp))
#
# k = 1
# for t in temp:
#     wt = glob(os.path.join(wav_dir, t+"*"))
#     c = np.random.choice(wt)
#     print(os.path.basename(c))
#     # print(os.path.join("./data/librspeech_other-4/val_wav", os.path.basename(c)))
#     os.rename(c, os.path.join("./data/librspeech_other-4/test_wav", os.path.basename(c)))
#     k += 1
#     if k >= 501:
#         break









