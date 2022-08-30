# coding=utf-8
"""
对音频数据的几种数据转换方式的实现
"""
import numpy as np
from scipy.fftpack import fft, ifft
import librosa
import scipy.io.wavfile as wav
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def read_wav(audio_path):
    '''使用scipy库下的方法读取音频文件wav,返回时间序列y（数据类型和声道数由文件本身决定）和采样率sr'''
    sr, y = wav.read(filename=audio_path)  # 读取音频文件，返回音频采样率和时间序列
    return y, sr


def write_wav(y, sr, save_path):
    '''使用scipy库下的方法，将时间序列保存为wav格式，y是音频时间序列，sr是采样率，save_path="***.wav"'''
    wav.write(filename=save_path, rate=sr, data=np.array(np.clip(np.round(y), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))


# ————————————————————————————————DFT和逆DFT——————————————————————————————————————————————————————————————————————
def dft(audio, sr, n=None):
    '''对输入音频数据进行DFT（离散傅里叶变换）
       输入参数：
       audio: 读取的音频时间序列数据；
       sr: 是音频文件的采样率。
    返回DFT后的频率数组及相应的幅值和音频文件的采样率。
    '''
    if (n == None):
        n = audio.shape[0]  # 音频时间序列的长度

    y = np.fft.rfft(a=audio, n=n)  # 离散傅里叶变换（DFT），返回一个长度为len(audio)/2+1的复数数组
    #     print(y[:10])
    f = np.fft.fftfreq(n=n, d=1 / sr)[
        :n // 2 + 1]  # 返回DFT后每个值对应的频率； 即 k*sr/n = k/d*n，k表示时间序列的索引值；(0, sr/n, 2*sr/n, ..., sr/2)
    f[-1] = f[-1] * -1
    return f, y, sr  # 返回DFT后的频率数组及相应的幅值和音频文件的采样率


def idft(y, n=None):
    '''逆DFT变换，是指将DFT得到的复数数组重新变换为音频时间序列的形式'''
    # y是对音频进行DFT的结果，n是DFT时设置的值；
    if (n == None):
        input = np.fft.irfft(y)
    else:
        input = np.fft.irfft(y, n)
    input = np.array(np.clip(np.round(input), -2 ** 15, 2 ** 15 - 1), dtype=np.int16)
    return input


# audio, sr = read_wav("sample.wav")
# print(audio.shape)
# print(audio[1000:1010])
# f, y, sr = dft(audio, sr) 
# y_inv = idft(y)
# print(y_inv.shape)
# print(y_inv[1000:1010])
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————


# ************************STFT和逆STFT*************************************************************************
def stft(audio):
    """对读取到的音频数据进行短时傅里叶变换（STFT），返回变换后的数组以及音频时间序列的长度"""
    n = audio.shape[0]  # 音频序列的长度
    n_fft = 2048  # 设置傅里叶变换前，将音频数据分成许多帧的每一帧的长度2048
    audio_pad = librosa.util.fix_length(audio, n + n_fft // 2)  # 为了方便STFT，对音频数据进行末尾0填充
    y = librosa.stft(audio_pad.astype(float), n_fft=n_fft)  # stft变换
    #     print(y.shape)                         # shape(1025, 102)，1025=n_fft/2+1，表示经STFT后，每帧的长度；102表示帧数
    return y, n


def istft(y, n):  # n指音频时间序列的长度
    '''逆STFT变换'''
    y_inv = librosa.istft(y, length=n)
    y_inv = np.array(np.clip(np.round(y_inv), -2 ** 15, 2 ** 15 - 1), dtype=np.int16)
    return y_inv


# audio, sr = read_wav("sample.wav")
# print(audio.shape)
# print(audio[1000:1010])
# y, n = stft(audio) 
# y_inv = istft(y, n)
# print(y_inv.shape)
# print(y_inv[1000:1010])
# ********************************************************************************************************


# ______________________奇异谱分析法(SSA)______________________________________________________________

def ssa(data, L, res):  # （注：所有序列都是按从0开始算的）
    '''奇异谱分析SSA的实现，参考自：https://blog.csdn.net/u012947501/article/details/84999765
    包括对音频数据进行奇异值分解(SVD)和重构两部分
    参数：
    data； 音频数据；
    L: 表示选取的窗口长度
    res: 表示保留下的特征值及特征向量的数量，用于重构，注意，res取值在(0, L)之间
    返回：重构的音频数据。

    注意：音频数据太长，即使1s长的音频都有1几万的长度，直接对其进行SSA操作，占用内存过大，容易出现内存不足的情况，
        建议先将音频数据分帧，然后再对每一帧进行SSA操作，最后再粘连。
    '''
    ##嵌入：构建轨迹矩阵
    N = data.shape[0]  # 时间序列的长度
    #     L = N // 10                                             # 选取适当窗口长度，取在[2, N//3]之间，是轨迹矩阵的行数
    K = N - L + 1  # K是构建的轨迹矩阵的列数
    X = np.zeros((L, K))
    for k in range(K):  # 构建轨迹矩阵X
        X[0:L, k] = data[k:k + L]

    ##SVD分解:  （注，SVD分解有两种python实现方法，一种是先特征值分解，得到特征向量和特征值，再求右奇异矩阵； 另一种是直接求得奇异值分解）
    # 参考自：https://www.cnblogs.com/endlesscoding/p/10058532.html
    # 方法一： 先求特征值和特征向量，再求右奇异矩阵
    S = np.dot(X, X.T)  # X * X.T
    s, u = np.linalg.eig(S)  # 返回矩阵S的特征值和特征向量
    index_sort = np.argsort(s)[::-1]  # 返回特征值的降序排序后对应的索引值排序
    u_ = u[:, index_sort]
    s_ = s[index_sort]  # 按降序后的索引值为特征值和特征向量分别排序; （注： 特征向量的每一列对应于一个特征值，而非每一行）
    #     s_inv = np.linalg.inv(np.diag(np.sqrt(s_)))               # 计算奇异值矩阵的逆
    #     v_ = s_inv.dot(u_.T).dot(X)                               # 计算右奇异矩阵
    v_ = (u_.T).dot(X)  # 若是用于奇异谱分析，这里就不用乘s_inv（即奇异值矩阵的逆），因为重构计算rca时需要再次乘以np.sqrt(s_)（即奇异值矩阵），所以这里不乘后面也就不用乘

    #     #方法二：
    #     u, s, vh = np.linalg.svd(X)                             # 这里的u、s和vh分别代表左奇异矩阵、奇异值矩阵和右奇异矩阵

    ##分组：在信号处理领域，通常认为前面r个较大的奇异值能够反应信号的主要能量； 所以在这儿取出前res个奇异值对应的特征向量
    rca = u_[:, :res].dot(v_[:res, :])  # shape(L,K)

    ##对交平均化重构信号  
    y = np.zeros((N, 1))

    Lp = min(L, K)
    Kp = max(L, K)

    ##重构
    for k in range(0, Lp - 1):
        for m in range(0, k + 1):
            y[k] = y[k] + rca[m, k - m]
        y[k] = y[k] / (k + 1)

    for k in range(Lp - 1, Kp):
        for m in range(0, Lp):
            y[k] = y[k] + rca[m, k - m]
        y[k] = y[k] / Lp

    for k in range(Kp, N):  # Kp = N - (Lp-1)
        x = N - k
        for m in range(x, 0, -1):
            y[k] = y[k] + rca[Lp - m, Kp - x - 1 + m]
        y[k] = y[k] / x
    return y  # y是重构的时间序列


# _______________________________________________________________________________________________


# ——————————————————————————————对音频数据分帧，以及复原——————————————————————————————————————————————

def seq_to_frame(y, frame_length=2048, hop_length=None):
    """
    对一维的音频数据进行分帧处理，返回分帧的数据。
    参数：
    y：音频数据；
    frame_length： 设置的每一帧的长度；
    hop_length； 设置相邻两帧开始位置相差的长度；
    返回：分帧后的数组，形状shape(帧数, frame_length)
    """
    n = y.shape[0]  # 音频数据的长度
    if hop_length is None:
        hop_length = int(frame_length // 4)
    pre_frame_num = int(np.ceil((n - frame_length) / hop_length + 1))  # 预计的帧的数量
    pre_y_len = hop_length * (pre_frame_num - 1) + frame_length  # 预计的音频数据的长度（≥n）
    y_pad = np.pad(y, (0, pre_y_len - n), mode='constant', constant_values=0)  # 将原始音频数据y填充到预计的长度 ，用0填充
    # 对填充的音频数据y_pad进行分帧
    yy = []
    for i in range(0, pre_y_len - frame_length + 1, hop_length):
        yy.append(y_pad[i:i + frame_length])
    y_frames = np.array(yy)  # 分帧后的结果，shape(帧数, frame_length)
    print("yzn: ", y_frames.shape)

    return y_frames


def frame_to_seq(y_frames, hop_length=None, org_len=None):
    """
    将分帧的音频数据还原回去。
    参数：
    y_frames：分帧的矩阵数据；
    hop_length：前后两帧初始位置的距离，要与分帧时的hop_length对应上；
    n： 表示音频时间序列的原始长度。
    返回：原始音频数据
    """
    frame_num, frame_length = y_frames.shape  # 帧数，帧长
    if hop_length is None:
        hop_length = int(frame_length // 4)

    # 将分帧后的数组返回为原始音频序列
    xx = []
    for i in range(frame_num):
        if i == (frame_num - 1):
            xx.append(y_frames[i, :])
        else:
            xx.append(y_frames[i, :hop_length])
    x = np.concatenate(xx)

    if org_len is not None:  # 若是给了音频序列原始长度，在还原其后，需要截掉分帧时填充的部分
        x = x[:org_len]
    return x


# x, sr = read_wav("./data/org_wav/A CIRCLE OF A FEW HUNDRED FEET IN CIRCUMFERENCE WAS DRAWN AND EACH OF THE PARTY TOOK A SEGMENT FOR HIS PORTION.wav")
# print(x[-10:])
# print(x.shape)
# n = x.shape[0]
# frame_length = 16384
#
# y_frames = seq_to_frame(x, frame_length=frame_length)
# print("yzn: ", y_frames.shape)
# print("yzn: ", y_frames[10, :10])
# xx = frame_to_seq(y_frames, org_len=n)
# print(xx[-10:])
# print(xx.shape)
#   ____________________________________________________________________________________________________________________


import tensorflow as tf


def tf_seq_to_frame(y, frame_length=2048, hop_length=None):
    """
    :param y: 音频数据，tensor，shape（batch_size, 音频长, 1）
    :param frame_length: 设置的每一帧的长度
    :param hop_length: 相邻两帧的首位置的间距
    :return: 分帧后的结果，shape(bs, 帧数,帧长, 1)，tensor
    """
    n = y.get_shape().as_list()[1]  # 音频数据的长度
    if hop_length is None:
        hop_length = int(frame_length // 4)
    pre_frame_num = int(np.ceil((n - frame_length) / hop_length + 1))  # 预计的帧的数量
    pre_y_len = hop_length * (pre_frame_num - 1) + frame_length  # 预计的音频数据的长度（≥n）
    y_pad = tf.pad(y, tf.constant([[0, 0], [0, pre_y_len - n], [0, 0]]), mode='constant', constant_values=0)  # 将原始音频数据y填充到预计的长度 ，用0填充
    # 对填充的音频数据y_pad进行分帧
    yy = []
    for i in range(0, pre_y_len - frame_length + 1, hop_length):
        yy.append(y_pad[:, i: i + frame_length, :])
    y_frames = tf.stack(values=yy, axis=1)  # 分帧后的结果，shape(帧数, frame_length)
    return y_frames


def tf_frame_to_seq(y_frames, hop_length=None, org_len=None):
    """
    :param y_frames: 音频的分帧后的tensor, shape(bs, 帧数，帧长，1)
    :param hop_length: 相邻帧的首位置间的距离
    :param org_len: 音频数据喂分帧前最初始的长度
    :return: 还原后的音频数据  shape(bs, 音频长, 1)
    """
    frame_num, frame_length = y_frames.get_shape().as_list()[1:3]  # 帧数，帧长
    if hop_length is None:
        hop_length = int(frame_length // 4)

    # 将分帧后的数组返回为原始音频序列
    for i in range(frame_num):
        x = y_frames[:, i, :hop_length, :] if i == 0 \
            else tf.concat([x, y_frames[:, i, :, :]], axis=1) if i == frame_num - 1 \
            else tf.concat([x, y_frames[:, i, :hop_length, :]], axis=1)

    if org_len is not None:  # 若是给了音频序列原始长度，在还原其后，需要截掉分帧时填充的部分
        x = x[:, :org_len, :]
    return x


if __name__ == "__main__":
    x, sr = read_wav("./data/org_wav/A CIRCLE OF A FEW HUNDRED FEET IN CIRCUMFERENCE WAS DRAWN AND EACH OF THE PARTY TOOK A SEGMENT FOR HIS PORTION.wav")
    y = tf.constant(value=x, dtype=tf.int32, shape=[1, len(x), 1])
    org_len = y.get_shape().as_list()[1]
    y_frames = tf_seq_to_frame(y, frame_length=16384)
    y_restore = tf_frame_to_seq(y_frames, org_len=org_len)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print(y_frames.shape)
        print(y_restore.shape)

        print(x[-10:])
        print(sess.run(y_restore)[0, -10:, 0])
