import numpy as np
import librosa


# 计算信噪比
def SNR_singlech(clean_file, original_file):
    clean, clean_fs = librosa.load(clean_file, sr=None, mono=True)  # 导入干净语音
    ori, ori_fs = librosa.load(original_file, sr=None, mono=True)  # 导入原始语音
    length = min(len(clean), len(ori))
    est_noise = ori[:length] - clean[:length]  # 计算噪声语音

    # 计算信噪比
    SNR = 10 * np.log10((np.sum(clean ** 2)) / (np.sum(est_noise ** 2)))
    print(SNR)


SNR_singlech('/media/ps/data/gxy/Adv_audio/result/newsnr/12-24-15-06_wgan-gp/train/org_audio/60000_0_eight seven two seven.wav', '/media/ps/data/gxy/Adv_audio/result/newsnr/12-24-15-06_wgan-gp/train/adv_audio/60000_0_eight seven two seven.wav')