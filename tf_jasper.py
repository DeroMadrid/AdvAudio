"""
根据OpenSeq2Seq特征提取的代码进行修改
open_seq2seq/data/speech2text/speech_utils.py get_speech_features_librosa
采用的特征提取方式为logfbank

一些关于jasper的参数记录
"num_audio_features" : 64,
"input_type": "logfbank",
"vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
"norm_per_feature" : True,
"window" : "hanning",
"precompute_nel_basis": True,
"sample_freq" :16000,
"pad_to": 16,
"dither": 1e-5,
"backend":"librosa"
"""
import numpy as np
import tensorflow as tf
import sys
import os
import io
import librosa
import scipy
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sys.path.append("DeepSpeech")

"""
mel_basis应该是8，64，257
fft应该是8，***，257

num_fft:512
window_stride:0.01
window_size:0.02
hop_length:160
win_length:320
"""

# batchsize这里有点问题 可能会造成tensor形状出现问题
# 传入的音频数据处理可能还有点问题 和OpenSeq2Seq的处理可能会不一样
def compute_logfbank(audio, **kwargs):
    """
    计算所给audio波形的logfbank特征。用TF实现的，是可微的。
    :param audio: shape(bs, 音频长)
    :return: 返回logfbank特征
    """
    batch_size, size = tf.shape(audio)[0], audio.get_shape().as_list()[1]
    audio = tf.cast(audio, tf.float32)  # shape(bs, size)， size即音频长
    audio += 1e-5 * tf.random_normal(shape=audio.shape)

    # 采样率为16000
    audio_duration = size * 1.0 / 16000.

    # 1. 预加强，一个高通滤波器
    # 对于每一个音频数据（向量），第一项不变，后面的每一项减去其前一项*0.97
    # audio = tf.concat(values=(audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1],
    #                           tf.zeros((batch_size, 512), dtype=tf.float32)), axis=1)  # shape(bs, size+512)
    audio = tf.concat(values=(audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1]), axis=1)  # shape(bs, size)

    # 2. 由于语音信号是短时平稳信号，所以需要分帧，一般是20-40ms为一帧。
    # Pad the time series so that frames are centered
    # 填充int(n_fft // 2) n_fft=512
    # audio = tf.pad(audio, paddings=[0, 256], mode='REFLECT')
    # Window the time series.
    # 对每一个音频（向量），进行分帧，每一‘帧长’512，帧与帧首位置之间的偏移是160，即帧与帧重叠部分为512-160
    # Pad the time series so that frames are centered first
    padding = tf.constant([[0, 0], [int(512 // 2), int(512 // 2)]])
    audio = tf.pad(audio, padding, mode='reflect')
    windowed = tf.stack([audio[:, j: j + 512] for j in range(0, size - 160, 160)], 1)  # shape(batch_size, 帧数, 帧长)

    # 2.1 加窗，为每一帧乘以一个汉明窗
    # window和librosa中可能有所出入
    window = np.hanning(M=320)  # M值与上面的帧长对应
    # window = scipy.signal.get_window(window=np.hanning, Nx=320, fftbins=True)

    # Pad the window out to n_fft size
    # window = util.pad_center(window, 512)
    # 这里将librosa.util.pad_center重新实现一遍
    n = window.shape[-1]
    lpad = int((512 - n) // 2)
    lengths = [(0, 0)] * window.ndim
    lengths[-1] = (lpad, int(512 - n - lpad))
    window = np.pad(window, lengths, mode='constant')

    # window = window.reshape((-1, 1))

    windowed = windowed * window  # shape(batch_size, 帧数，帧长）
    print("windowed:", windowed.shape)

    # 3. 通过对每一帧进行FFT（傅里叶变换）将其转化到频域
    ffted = tf.signal.rfft(windowed, [512])
    # 3.1 计算功率谱
    ffted = tf.square(tf.abs(ffted))  # shape(batch_size, 帧数，'帧长/2+1')
    ffted = tf.transpose(ffted, [0,2,1])

    # energy = tf.reduce_sum(ffted, axis=2) + np.finfo(float).eps  # shape(batch_size, 帧数）
    # 4. 计算mel滤波器组
    mel_basis = librosa.filters.mel(16000., 512, n_mels=64, fmin=0, fmax=int(16000. / 2))
    mel_basis = tf.tile(tf.expand_dims(mel_basis, axis=0), [batch_size,1,1])
    print("mel_basis:", mel_basis.shape)

    feat = tf.log(tf.matmul(mel_basis, ffted) + 1e-20)
    feat = tf.transpose(feat, [0, 2, 1])
    # print("feat:", feat)
    mean, variance = tf.nn.moments(feat, axes=1)
    feat = tf.transpose(feat, [1, 0, 2])
    print("mean:", mean.shape, "    variance:", variance.shape)
    feat = (feat - mean) / tf.sqrt(variance)

    # print("feat:",feat)
    feat = tf.transpose(feat, [1, 0, 2])
    # print("feat.T:", feat)
    return  audio_duration, feat


"""
后面部分和获取jasper字典相关
"""
def load_pre_existing_vocabulary(path, min_idx=0, read_chars=False):
  """Loads pre-existing vocabulary into memory.

  The vocabulary file should contain a token on each line with optional
  token count on the same line that will be ignored. Example::

    a 1234
    b 4321
    c 32342
    d
    e
    word 234

  Args:
    path (str): path to vocabulary.
    min_idx (int, optional): minimum id to assign for a token.
    read_chars (bool, optional): whether to read only the
        first symbol of the line.

  Returns:
     dict: vocabulary dictionary mapping tokens (chars/words) to int ids.
  """
  idx = min_idx
  vocab_dict = {}
  with io.open(path, newline='', encoding='utf-8') as f:
    for line in f:
      # ignoring empty lines
      if not line or line == '\n':
        continue
      if read_chars:
        token = line[0]
      else:
        token = line.rstrip().split('\t')[0]
      vocab_dict[token] = idx
      idx += 1
  return vocab_dict

char2idx = load_pre_existing_vocabulary("OpenSeq2Seq/open_seq2seq/test_utils/toy_speech_data/vocab.txt", read_chars=True,)
idx2char = {i: w for w, i in char2idx.items()}