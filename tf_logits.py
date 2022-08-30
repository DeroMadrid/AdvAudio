"""
主要是调用DeepSpeech模型，返回相应的logits输出（未经softmax归一化的输出）（get_logits函数）
"""
import numpy as np
import tensorflow as tf
import argparse
import scipy.io.wavfile as wav
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from glob import glob
import DeepSpeech
from share import toks, ds_preds, ds_preds_lm, ds_var
from digit_rectify import digit_lm

sys.path.append("DeepSpeech")


def mel_filterbanks(nfilt, sample_rate, NFFT):
    """
    计算filterbanks
    :param nfilt: 定义的滤波器个数
    :param sample_rate: 采样率
    :param NFFT: 帧长
    """
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # 将Hz转换为Mel
    # 我们要做nfilt个滤波器组，为此需要（nfilt+2）个点，这意味着在们需要low_freq_mel和high_freq_mel之间线性间隔nfilt个点
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 使得Mel scale间距相等
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # 将Mel转换回-Hz
    bins = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bins[m - 1])  # 左
        f_m = int(bins[m])  # 中
        f_m_plus = int(bins[m + 1])  # 右

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return fbank  # shape(nfilt, NFFT//2+1)


def compute_mfcc(audio, **kwargs):
    """
    计算所给audio波形的mfcc特征。它的实现方式和DeepSpeech里面的是一样的，但是是用TF实现的，是可微的。
    :param audio: shape(bs, 音频长)
    :return: 返回mfcc（梅尔频率倒谱系数）特征，具体实现可参考：https://www.cnblogs.com/LXP-Never/p/10918590.html
    """
    batch_size, size = audio.get_shape().as_list()
    audio = tf.cast(audio, tf.float32)  # shape(bs, size)， size即音频长

    # 1. 预加强，一个高通滤波器
    # 对于每一个音频数据（向量），第一项不变，后面的每一项减去其前一项*0.97，然后，再在其后填充512个0
    audio = tf.concat(values=(audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1],
                              np.zeros((batch_size, 512), dtype=np.float32)), axis=1)  # shape(bs, size+512)

    # 2. 由于语音信号是短时平稳信号，所以需要分帧，一般是20-40ms为一帧。
    # 对每一个音频（向量），进行分帧，每一‘帧长’512，帧与帧首位置之间的偏移是320，即帧与帧重叠部分为512-320=192
    windowed = tf.stack([audio[:, j: j + 512] for j in range(0, size - 320, 320)], 1)  # shape(batch_size, 帧数, 帧长)
    # 2.1 加窗，为每一帧乘以一个汉明窗
    window = np.hamming(M=512)  # M值与上面的帧长对应
    windowed = windowed * window  # shape(batch_size, 帧数，帧长）

    # 3. 通过对每一帧进行FFT（傅里叶变换）将其转化到频域
    ffted = tf.signal.rfft(windowed, [512])
    # 3.1 计算功率谱
    ffted = 1.0 / 512 * tf.square(tf.abs(ffted))  # shape(batch_size, 帧数，'帧长/2+1')

    energy = tf.reduce_sum(ffted, axis=2) + np.finfo(float).eps  # shape(batch_size, 帧数）
    # 4. 计算mel滤波器组
    filters = np.load("filterbanks.npy").T  # 这里的shape（257, 26），26表示选取了26个滤波器
    # filters1 = mel_filterbanks(nfilt=26, sample_rate=16000, NFFT=512).T   # 它等价于上面一行
    feat = tf.matmul(ffted, np.array([filters]*batch_size, dtype=np.float32)) + np.finfo(float).eps  # (bs, 帧数， 26）

    # 5. Take the DCT again, because why not。离散余弦变换
    feat = tf.log(feat)  # 自然对数
    feat = tf.signal.dct(input=feat, type=2, norm='ortho')[:, :, :26]  # shape(bs, 帧数, 26)

    # 6. Amplify high frequencies for some reason
    _, nframes, ncoeff = feat.get_shape().as_list()
    n = np.arange(ncoeff)
    lift = 1 + (22 / 2.) * np.sin(np.pi * n / 22)  # shape(26,)
    feat = lift * feat  # shape(bs, 帧数, 26)
    width = feat.get_shape().as_list()[1]  # 帧数

    # 7. And now stick the energy next to the features
    feat = tf.concat((tf.reshape(tf.log(energy), (-1, width, 1)), feat[:, :, 1:]), axis=2)  # shape(bs, 帧数， 26）
    return feat


def get_logits(new_input, first=[]):
    """
    计算所给波形（即音频数据）的logits。
    首先使用TF版的MFCC进行预处理，得到features，然后将所得的features输入deepspeech，得到其logits输出。
    :param new_input: shape(bs, 音频长），表示输入的数据
    :param first: 记录是否首次调用DeepSpeech
    """
    batch_size = new_input.get_shape()[0]  # bs

    # 1. 计算输入音频的MFCC（上面的实现mfcc的方法是可微的）。注：下面的9、26等数字等同deepspeech-0.4.1源码中的设置
    empty_context = np.zeros((batch_size, 9, 26), dtype=np.float32)  # shape(bs, 9, 26)
    new_input_to_mfcc = compute_mfcc(new_input)  # shape(bs, 帧数， 26），计算输入信号的mfcc特征

    features = tf.concat((empty_context, new_input_to_mfcc, empty_context), 1)  # shape(bs, 9+帧数+9， 26）

    # 2. We get to see 9 frames at a time to make our decision,
    # so concatenate them together.
    # 我们可以一次同时通过9个帧来做决策，所以9个帧为一组进行粘连
    # 9 (past) + 1 (present) + 9 (future)
    features = tf.reshape(features, [new_input.get_shape()[0], -1])  # shape(bs, '9+帧数+9'* 26)
    # 与分帧类似（帧长19*26，帧与帧首位置间距26），mfcc特征的一帧长是26.  shape(bs, 帧数, 19*26)
    features = tf.stack([features[:, j: j+19*26] for j in range(0, features.shape[1] - 19 * 26 + 1, 26)], 1)
    features = tf.reshape(features, [batch_size, -1, 19, 26])  # shape(bs, 帧数, 19, 26)

    # 帧数，shape(bs,)
    length = tf.constant(value=features.get_shape().as_list()[1], shape=(batch_size,), dtype=tf.int32)

    # 3. Finally we process it with DeepSpeech
    # We need to init DeepSpeech the first time we're called
    # 调用DeepSpeech，第一次调用时需要初始化DeepSpeech
    if not first:
        first.append(False)
        DeepSpeech.create_flags()
        tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
        DeepSpeech.initialize_globals()

    logits, _ = DeepSpeech.BiRNN(batch_x=features, seq_length=length, dropout=[0]*6)  # shape(帧数, bs, 29)

    return logits, length


def logits_test(audio_dir_, lm_):
    """
    用于测试上面的代码的运行结果
    :param audio_dir_: 需要转录的音频文件目录
    :param lm_: 若为True，表示使用语言模型
    """
    fps = glob(os.path.join(audio_dir_, "*"))
    audios = []
    audio_name = []
    for fp in fps:
        sr, audio = wav.read(fp)
        assert sr == 16000
        audios.append(audio)
        audio_name.append(os.path.basename(fp).split('.')[0])

    audios_maxlen = max(map(len, audios))
    audios = [np.pad(ad, (0, audios_maxlen - len(ad)), mode='constant', constant_values=0) for ad in audios]

    batch_size = 1

    # 下面的shape不能为None，否则会影响mfcc的计算
    input_tf = tf.placeholder(dtype=tf.float32, shape=[batch_size, audios_maxlen], name="audio_inputs")

    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        logits, seq_len = get_logits(input_tf)  # shape(帧数 ,bs, 29)
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False, beam_width=100)

    logits_lm = tf.transpose(tf.nn.softmax(logits), [1, 0, 2])

    ds_saver = tf.train.Saver(var_list=ds_var(tf.global_variables()))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        checkpoint = tf.train.get_checkpoint_state("./deepSpeech_model/ds-0.4.1-ckpt")
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(checkpoint_dir))
            exit(1)
        checkpoint_path = checkpoint.model_checkpoint_path
        ds_saver.restore(sess, checkpoint_path)

        loops = len(audios) // batch_size
        for i in range(loops):
            p1, p2 = i*batch_size, i*batch_size+batch_size
            audio_bs = np.array(audios[p1: p2], dtype=float)
            audio_name_bs = audio_name[p1: p2]
            logit_lm, decode, slen = sess.run([logits_lm, decoded[0], seq_len], feed_dict={input_tf: audio_bs})

            if not lm_:
                tran = ds_preds(decode)
                # tran = digit_lm(tran)   # 通过编辑距离矫正无效单词（仅适用于数字音频）
            else:
                # print("带有语言模型的解码结果：")
                tran = ds_preds_lm(logit_lm, slen)

            for i, j in zip(audio_name_bs, tran):
                print("{0}:   {1}".format(i, j))  # 输出其音频文件名:  音频转录


if __name__ == "__main__":
    lm = False  # 若为True，表示使用deepspeech的语言模型解码
    audio_dir = "./data/digit/uniform/9"  # 音频文件目录
    logits_test(audio_dir, lm)
