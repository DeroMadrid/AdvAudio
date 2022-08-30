# coding=utf-8
from scipy.io.wavfile import read, write
import numpy as np
import tensorflow as tf
import sys
from glob import glob
import librosa
from share import toks
import os
from tf_jasper import compute_logfbank, idx2char, char2idx
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def decode_audio(fp, fs=None, num_channels=1, normalize=False, fast_wav=False, audio_maxlen=65536):
    """
    读取一个音频文件，返回音频数据，其shape(音频长度, 声道数），数据类型是float32，取值范围[-1,1]。
    :param fp: 音频文件路径名
    :param fs: 采样率
    :param num_channels: 音频的声道数
    :param normalize: 是否标准化
    :param fast_wav: 若为True，使用scipy.io.wavfile.read读取音频文件（相对快一些）；否则应用librosa
    """
    if fast_wav:
        _fs, _wav = read(filename=fp)     # 采样率，音频时间序列
        if fs is not None and fs != _fs:  # scipy库下没有重采样的方法
            raise NotImplementedError('Scipy cannot resample audio.')
        if _wav.dtype == np.int16:
            _wav = _wav.astype(np.float32)
            _wav /= 32768. # 归一化 转换为[-1,1]之间
        elif _wav.dtype == np.float32:
            _wav = np.copy(_wav)
        else:
            raise NotImplementedError('Scipy cannot process atypical WAV files.')
    else:
        _wav, _fs = librosa.core.load(path=fp, sr=fs, mono=False)
        if _wav.ndim == 2:  # 若非单声道，交换其0、1维，使得声道数放在最后一维；
            _wav = np.swapaxes(_wav, 0, 1)

    assert _wav.dtype == np.float32  # 判断音频时间序列类型是否是float32，若不是，则出错退出

    # _wav统一转化为shape(nsamps, nch)
    if _wav.ndim == 1:          # 单声道
        nsamps = _wav.shape[0]  # 采样点数
        nch = 1                 # 声道数
    else:                       # 非单声道
        nsamps, nch = _wav.shape
    _wav = np.reshape(_wav, [nsamps, nch])

    # 将_wav的最后一维nch转化为指定的num_channels
    if nch != num_channels:
        if num_channels == 1:
            _wav = np.mean(_wav, axis=1, keepdims=True)
        elif nch == 1 and num_channels == 2:
            _wav = np.concatenate([_wav, _wav], axis=1)
        else:
            raise ValueError('Number of audio channels not equal to num specified')

    # 标准化_wav
    if normalize:
        factor = np.max(np.abs(_wav))
        if factor > 0:
            _wav /= factor

    # mask 和
    mask = np.array([1 if i < nsamps else 0 for i in range(audio_maxlen)], dtype=np.float32)
    max_length_freq = (audio_maxlen // 2 + 1) // 240 * 3
    length_freq = (nsamps // 2 + 1) // 240 * 3
    mask_freq = np.zeros([max_length_freq, 80], dtype=np.float32)
    mask_freq[:length_freq, :] = 1
    _wav = np.pad(_wav, pad_width=((0, audio_maxlen-nsamps), (0, 0)), mode='constant', constant_values=0)
    return _wav, mask, mask_freq


def get_audio_maxlen(fps):
    max_l = 0
    for fp in fps:
        wav = decode_audio(fp, fs=16000, num_channels=1, normalize=False, fast_wav=True)
        max_l = max(max_l, wav.shape[0])
    return max_l


def get_fps_tran(wav_dir, tran_txt):
    """
    读取音频文件和相应的转录文件
    :param wav_dir: 音频文件所在目录
    :param tran_txt: 存放转录的文件，每一行的格式： ‘名字:原转录或名字:ds转录:lv转录’
    :return: 音频文件路径名列表，deepspeech转录序列，deepspeech转录，lingvo转录，音频长度最大值，deepspeech转录长度最大值
    """
    fps = glob(os.path.join(wav_dir, '*'))     # 音频文件名组成的列表

    # 读取转录文件，根据音频文件名获取相应的转录
    f_tran = open(tran_txt, 'r')
    f_tran_read = f_tran.readlines()
    f_tran.close()
    name_tran_d = {}
    for ftr in f_tran_read:
        # name, org_tran, ds_org_tran, lv_org_tran = ftr.strip('\n').split(":")
        # name_tran_d[name] = [org_tran, ds_org_tran.lower(), lv_org_tran.lower()]
        # name, org_tran, ds_org_tran, lv_org_tran, js_org_tran, wl_org_tran = ftr.strip('\n').split(":")
        # name_tran_d[name] = [org_tran, ds_org_tran.lower(), lv_org_tran.lower(), js_org_tran.lower(),
        #                     wl_org_tran.lower()]
        name, org_tran, ds_org_tran, lv_org_tran, js_org_tran = ftr.strip('\n').split(":")
        name_tran_d[name] = [org_tran.lower(), ds_org_tran.lower(), lv_org_tran.lower(), js_org_tran.lower()]

    # 根据音频名字，将其对应的各个转录放入相应的list
    # org_trans, ds_org_trans, lv_org_trans = [], [], []
    # org_trans, ds_org_trans, lv_org_trans, js_org_trans, wl_org_trans = [], [], [], [], []
    org_trans, ds_org_trans, lv_org_trans, js_org_trans = [], [], [], []
    for fp in fps:
        fname = os.path.basename(fp).split('.')[0]      # 音频文件名
        org_trans.append(name_tran_d[fname][0])         # 原转录或文件名
        ds_org_trans.append(name_tran_d[fname][1])      # deepspeech原转录
        lv_org_trans.append(name_tran_d[fname][2])      # lingvo原转录
        js_org_trans.append(name_tran_d[fname][3])  # jasper原转录
        # wl_org_trans.append(name_tran_d[fname][4])  # wav2letter原转录

    # deepspeech原转录的序列化表示，并将它们填充到统一长度
    ds_org_tran_maxlen = max(map(len, ds_org_trans))
    ds_org_seqs = [list(np.pad([list(toks()).index(k) for k in dot], (0, ds_org_tran_maxlen-len(dot)),
                                   mode="constant", constant_values=0)) for dot in ds_org_trans]
    # print(np.array(ds_org_seqs).shape)

    # 此处若是需要js和wl的原转录的序列化表示的话在这里填充
    js_org_tran_maxlen = max(map(len, js_org_trans))
    js_org_seqs = [list(np.pad([char2idx[k] for k in dot], (0, js_org_tran_maxlen - len(dot)),
                               mode="constant", constant_values=0)) for dot in js_org_trans]
    # print(np.array(js_org_seqs).shape)

    # return fps, ds_org_seqs, ds_org_trans, lv_org_trans, js_org_seqs, js_org_tran, wl_org_tran, \
    #        ds_org_tran_maxlen, js_org_tran_maxlen
    return fps, ds_org_seqs, ds_org_trans, lv_org_trans, js_org_seqs, js_org_trans,\
           ds_org_tran_maxlen, js_org_tran_maxlen, org_trans
    # return fps, ds_org_seqs, ds_org_trans, lv_org_trans, js_org_tran, ds_org_tran_maxlen


def decode_extract_and_batch(wav_dir, tran_txt, shuffle=False, repeat=False, batch_size=1, decode_fs=16000,
                             decode_num_channels=1,
                             decode_normalize=True,
                             decode_fast_wav=True,
                             decode_parallel_calls=4,
                             prefetch_gpu_num=0,
                             prefetch_size=1):
    """
    对所有音频文件读取、shape和格式转化、分batch，返回一个dataset
    :param wav_dir: 音频文件目录
    :param tran_txt: 存放转录的文件，格式： '名字:原转录或名字:ds转录:lv转录'
    :param shuffle: 若为True，表示随机打乱数据集
    :param repeat: 若为True，表示重复遍历所有数据
    :param batch_size: 一个batch的大小
    :param decode_fs: 采样率
    :param decode_num_channels: 音频数据的最后一维大小，默认为1
    :param decode_normalize: 若为True，表示对音频数据标准化处理
    :param decode_fast_wav: 若为True，表示以scipy.io.wavfile.read方法读取音频文件（相比librosa快一些）
    :param decode_parallel_calls: dataset.map操作时，decode_parallel_calls个数据同时被操作
    :param prefetch_gpu_num: 所要使用的gpu号
    :param prefetch_size: 预读取的batch数，一般为1
    :return: 一个dataset
    """
    # fps, ds_org_seqs, ds_org_trans, lv_org_trans, js_org_tran, wl_org_tran, \
    # ds_tran_maxlen = get_fps_tran(wav_dir, tran_txt)
    # fps, ds_org_seqs, ds_org_trans, lv_org_trans, js_org_tran, \
    #     ds_tran_maxlen = get_fps_tran(wav_dir, tran_txt)
    fps, ds_org_seqs, ds_org_trans, lv_org_trans, js_org_seqs, js_org_trans, \
    ds_tran_maxlen, js_tran_maxlen, org_trans = get_fps_tran(wav_dir, tran_txt)

    audio_data_maxlen = 65536
    max_length_freq = (audio_data_maxlen // 2 + 1) // 240 * 3

    # 参考：https://tensorflow.juejin.im/programmers_guide/datasets.html
    # dataset = tf.data.Dataset.from_tensor_slices((fps, ds_org_seqs, ds_org_trans, lv_org_trans, js_org_tran, wl_org_tran))
    # dataset = tf.data.Dataset.from_tensor_slices((fps, ds_org_seqs, ds_org_trans, lv_org_trans, js_org_seqs, js_org_tran))
    dataset = tf.data.Dataset.from_tensor_slices((fps, ds_org_seqs, ds_org_trans, lv_org_trans, js_org_seqs, js_org_trans, org_trans))

    # buffer_size一般设置大于等于总的item数。表示从总的item中按顺序从头开始抽取buffer_size个item放入buffer，做数据处理，
    # 每次从buffer中随机拿走batch_size个item训练模型，每从buffer中取走一个item，就立即从总的item中顺延着取1个item补到buffer中。
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(fps))  # 训练时为len(fps)，验证时可为1

    if repeat:
        dataset = dataset.repeat()  # 训练时为None（由于固定了其为int型，所以将其写成一个极大值），验证时可为1

    # 先将单个音频文件的解码（读取）过程转化为一个匿名函数;
    # 再用py_func将上面的匿名函数（python函数）封装为为tensorflow graph中的一个操作op，这样，我们就可以将一个tensor传入该函数，
    # 执行numpy的操作，返回tensor输出。
    # set_shape更新tensor的shape，一般补充shape为None的情况，且不能跨阶数改变形状。
    # def _decode_audio_shaped(fp, ds_org_seq_, ds_org_tran_, lv_org_tran_, js_org_tran_, wl_org_tran_):
    def _decode_audio_shaped(fp, ds_org_seq_, ds_org_tran_, lv_org_tran_, js_org_seq_, js_org_tran_, org_trans):
        _decode_audio_closure = lambda _fp: decode_audio(
            _fp,
            fs=decode_fs,
            num_channels=decode_num_channels,
            normalize=decode_normalize,
            fast_wav=decode_fast_wav)
        audio, mask, mask_freq = tf.py_func(func=_decode_audio_closure, inp=[fp],
                                            Tout=[tf.float32, tf.float32, tf.float32],
                                            stateful=False,
                                            name="decode_audio")
        audio.set_shape([None, decode_num_channels])
        # return audio, ds_org_seq_, ds_org_tran_, lv_org_tran_, js_org_tran_, wl_org_tran_, mask, mask_freq
        return audio, ds_org_seq_, ds_org_tran_, lv_org_tran_, js_org_seq_, js_org_tran_, mask, mask_freq, org_trans

    # 对dataset中的每一item执行_decode_audio_shaped函数
    # 注：num_parallel_calls的值设置不要大于自己机器的cpu内核数，表示并行调用map_func指定的函数。
    dataset = dataset.map(map_func=_decode_audio_shaped, num_parallel_calls=decode_parallel_calls)

    # 制作batch。drop_remainder为True表示将分批后多余的部分丢弃，否则将剩余的部分以小于batchsize的大小组一个batch
    # dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=([audio_data_maxlen, decode_num_channels], [ds_tran_maxlen], [], [],
                                                  [js_tran_maxlen], [], [audio_data_maxlen], [max_length_freq, 80], []),
                                   padding_values=(tf.cast(0, tf.float32), tf.cast(0, tf.int32), tf.cast("", tf.string),
                                                   tf.cast("", tf.string), tf.cast(0, tf.int32), tf.cast("", tf.string),
                                                   tf.cast(0, tf.float32), tf.cast(0, tf.float32), tf.cast("", tf.string)),
                                   drop_remainder=True)

    # print(dataset.output_shapes)
    # print(dataset.output_types)

    # 若是给了gpu号，则使用该gpu
    # dataset.prefetch：用于在请求输入数据集之前从输入数据集中预提取元素，从而减少了GPU闲置时间，
    # 其参数buffer_size表示预读取的batch数，一般设置为1
    if prefetch_size is not None:
        dataset = dataset.prefetch(buffer_size=prefetch_size)
        if prefetch_gpu_num is not None and prefetch_gpu_num >= 0:
            dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/device:GPU:{}'.format(prefetch_gpu_num)))
    return dataset


def get_iterator(handle, train_bs, test_bs, train_wav_dir, train_tran_txt,
                 test_wav_dir=None, test_tran_txt=None):
    
    # 返回读取训练集和验证集的迭代器。参数分别是训练集和验证集的音频文件目录、转录txt文件 和 用于选择训练还是验证的句柄

    with tf.name_scope("decode"):
        train_dataset = decode_extract_and_batch(train_wav_dir, train_tran_txt, shuffle=True, repeat=True,
                                                    batch_size=train_bs)
        test_dataset = decode_extract_and_batch(test_wav_dir, test_tran_txt, batch_size=test_bs,
                                                      prefetch_gpu_num=-1)

        # make_initializable_iterator()和make_one_shot_iterator()的区别在于，前者需要在Session创建后初始化，且可以多次初始化，
        # 每初始化一次，就从头读取数据；后者不需要初始化，只有一次从头开始读取数据的机会，不能再次或多次重新从头开始读取数据。
        train_iterator = train_dataset.make_one_shot_iterator()
        test_iterator = test_dataset.make_initializable_iterator()
        iterator = tf.data.Iterator.from_string_handle(string_handle=handle, output_types=train_dataset.output_types,
                                                       output_shapes=train_dataset.output_shapes)
    return iterator, train_iterator, test_iterator



"""
if __name__ == "__main__":
    
    # 尽可能在跑代码之前，先将所有音频文件统一长度，所有转录也统一长度。
    
    # training_wav_dir = "./data/digit/uniform/1"  # 训练集音频文件所在目录(仅是用于测试代码的正确性）
    # validation_wav_dir = "./data/digit/uniform/2"
    training_wav_dir = "./data/recaptchaV2/train/wav"
    validation_wav_dir = "./data/recaptchaV2/val/wav"
    training_tran_txt = 'data/recaptchaV2/training/tran_d_l_p.txt'  # 训练集转录文件
    validation_tran_txt = 'data/recaptchaV2/training/tran_d_l_p.txt'

    # training_tran_txt = './data/digit/tran_d_l_p.txt'  # 训练集转录文件
    # validation_tran_txt = './data/digit/tran_d_l_p.txt'  # 验证集转录文件

    train_batch_size = 1
    validation_batch_size = 1

    handle = tf.placeholder(dtype=tf.string, shape=[], name='trainOrValidation')

    with tf.name_scope("loader"):
        param = [handle, train_batch_size, validation_batch_size, training_wav_dir, training_tran_txt,
                 validation_wav_dir, validation_tran_txt]
        iterator, training_iterator, validation_iterator = get_iterator(*param)

    next_element = iterator.get_next()

    x = next_element[0]  # 原始音频数据值，shape(batch_size, 音频长, nch)
    ds_org_seqs = next_element[1]  # deepspeech原转录的数字序列形式， shape(batch_size, seq_maxlen)
    ds_org_trans = next_element[2]  # deepspeech原转录，shape(batch_size,)
    lv_org_trans = next_element[3]  # lingvo原转录，shape(batch_size,)
    js_org_trans = next_element[4]  # jasper原转录
    wl_org_trans = next_element[5]  # wav2letter+原转录
    tf_masks = tf.expand_dims(next_element[6], -1)
    tf_masks_freq = next_element[7]
    # x.set_shape([None, 65536, 1])
    print(x.shape)
    print(tf_masks.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        writer = tf.summary.FileWriter(logdir="./", graph=sess.graph)

        i = 0
        while i < 1:
            print("训练集测试：")
            a, b, c, d, e, f = sess.run([x, ds_org_seqs, ds_org_trans, lv_org_trans, tf_masks, tf_masks_freq],
                                  feed_dict={handle: training_handle})
            print(a.shape)
            print(c)
            print(d)
            print(e.shape)
            print(f.shape)
            # print(np.sum((a*e) == a) == 65536)
            print("-----------------------------------------------")
            sess.run(validation_iterator.initializer)
            print("验证集遍历：")
            a, b, c, d, e, f = sess.run([x, ds_org_seqs, ds_org_trans, lv_org_trans, tf_masks, tf_masks_freq],
                                        feed_dict={handle: validation_handle})
            print(a.shape)
            print(c)
            print(d)
            print(e.shape)
            print(f.shape)
            print('\n')
            # while True:
            #     try:
            #         a, b, c, d, e, f = sess.run([x, ds_org_seqs, ds_org_trans, lv_org_trans, tf_masks, tf_masks_freq],
            #                               feed_dict={handle: validation_handle})
            #         print(c)
            #         print(d)
            #         print(e)
            #         print(f)
            #         print('\n')
            #     except tf.errors.OutOfRangeError:
            #         break
            i += 1
"""


