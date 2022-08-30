# coding=utf-8
"""调用deepspeech的checkpoint的是使用方法"""
import numpy as np
import tensorflow as tf
from glob import glob
import scipy.io.wavfile as wav

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import sys
import pydub
import struct
from share import toks, ds_preds, ds_preds_lm, ds_var
from tf_logits import get_logits
from digit_rectify import digit_lm


sys.path.append("DeepSpeech")


def call_target(audios_, tgts_, tran_d_, audio_data_maxlen_, lm_=True, tran_save_txt_=None):
    batch_size = 1

    input_tf = tf.placeholder(dtype=tf.float32, shape=[batch_size, audio_data_maxlen_], name="audio_inputs")
    tgt_tf = tf.placeholder(dtype=tf.string, shape=[batch_size], name="orgTran_or_filename")

    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        logits, seq_len = get_logits(input_tf)  # shape(帧数, bs, 29)

    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False, beam_width=100)

    logits_lm = tf.transpose(tf.nn.softmax(logits), [1, 0, 2])

    ds_saver = tf.train.Saver(ds_var(tf.global_variables()))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)

    with tf.Session(config=config) as sess:
        latest_ckpt_fp = tf.train.latest_checkpoint("./deepSpeech_model/ds-0.4.1-ckpt")
        ds_saver.restore(sess, latest_ckpt_fp)

        loops = len(audios_) // batch_size

        for i in range(loops):
            p1, p2 = i*batch_size, (i+1)*batch_size
            audio_bs = np.array(audios_[p1: p2], dtype=float)
            tgt_bs = tgts_[p1: p2]
            tgt_np, logit_lm, decode, slen = sess.run([tgt_tf, logits_lm, decoded[0], seq_len],
                                                      feed_dict={input_tf: audio_bs, tgt_tf: tgt_bs})

            if not lm_:
                res = ds_preds(decode)

            else:
                res = ds_preds_lm(logit_lm, slen)

            # res = digit_lm(res)
            if tran_save_txt_ is None:  # 如果仅是预测结果，就选择它
                for r in res:
                    print(r)
            else:
                fp = open(tran_save_txt_, mode='a')  # 将原始转录文本和deepspeech的转录文本保存到该txt文件中
                for r in range(len(res)):
                    if isinstance(tgt_np[r], bytes):
                        tgt_np[r] = str(tgt_np[r], encoding='utf-8')
                    if tran_d_ is not None:
                        print("{0}:\t{1}:\t{2}".format(tgt_np[r], tran_d_[tgt_np[r]], res[r]))
                        fp.write("{0}:{1}:{2}\n".format(tgt_np[r], tran_d_[tgt_np[r]], res[r]))
                    else:
                        print("{0}:\t{1}".format(tgt_np[r], res[r]))
                        fp.write("{0}:{1}\n".format(tgt_np[r], res[r]))
                fp.close()
    return


def ds_main(audio_dir_, tran_txt_=None, tran_d_txt_=None, lm_=False):
    """
    :param audio_dir_: 需要转录的音频文件目录
    :param tran_txt_: 初始的转录保存文件，每行格式：'音频文件名:音频文件名or原始转录'
    :param tran_d_txt_: 保存deepspeech的转录结果，每行格式：'音频文家名:音频文件名or原转录:deepspeech转录'
    :param lm_: 若为True，表示使用语言模型
    """
    if tran_txt_ is not None:
        f = open(tran_txt_, mode='r')
        lines = f.readlines()
        f.close()

        tran_d = {}
        for line in lines:
            name, org_tran = line.strip().split(':')
            tran_d[name] = org_tran
    else:
        tran_d = None

    audios = []
    tgts = []

    audio_path_list = glob(os.path.join(audio_dir_, '*'))
    for audio_path in audio_path_list:
        fname = os.path.basename(audio_path).split(".")[0]
        # print("fname:", fname)
        sr, audio = wav.read(audio_path)
        audios.append(audio)
        tgts.append(fname)
    # print("audios:", audios)

    audio_data_maxlen = max(map(len, audios))
    audios = [np.pad(i, pad_width=(0, audio_data_maxlen - len(i)), mode="constant") for i in audios]

    call_target(audios, tgts, tran_d, audio_data_maxlen, lm_=lm_, tran_save_txt_=tran_d_txt_)


if __name__ == "__main__":
    # tran_txt = './data/librspeech_other-4/tran.txt'  # 每一行格式，'音频文件名:音频文件名'，说明文件中第4）步所得的txt文件
    # audio_dir = "./data/librspeech_other-4/train_wav"  # 音频文件目录
    # tran_d_txt = "./data/librspeech_other-4/tran_d.txt"  # 得到的新的文件格式，'音频文件名子:音频文件名字:deepspeech转录'
    # tran_txt = '/media/ps/data/YZN/Adv_audio/data/recaptchaV2/recaptcha5k/tranval.txt'  # 每一行格式，'音频文件名:音频文件名'，说明文件中第4）步所得的txt文件
    # audio_dir = "/media/ps/data/YZN/Adv_audio/data/recaptchaV2/recaptcha5k/val"  # 音频文件目录
    # tran_d_txt = "/media/ps/data/YZN/Adv_audio/data/recaptchaV2/recaptcha5k/tranval_d.txt"  # 得到的新的文件格式，'音频文件名子:音频文件名字:deepspeech转录'

    # tran_txt = '/media/ps/data/gxy/Adv_audio/data/recaptchaV2/val/tran.txt'  # 每一行格式，'音频文件名:音频文件名'，说明文件中第4）步所得的txt文件
    # audio_dir = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/val/wav"  # 音频文件目录
    # tran_d_txt = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/val/tran_d.txt"  # 得到的新的文件格式，'音频文件名子:音频文件名字:deepspeech转录'

    # tran_txt = '/media/ps/data/gxy/Adv_audio/data/digit/tran.txt'  # 每一行格式，'音频文件名:音频文件名'，说明文件中第4）步所得的txt文件
    # audio_dir = "/media/ps/data/gxy/Adv_audio/data/digit/uniform"  # 音频文件目录
    # tran_d_txt = "/media/ps/data/gxy/Adv_audio/data/digit/tran_d.txt"  # 得到的新的文件格式，'音频文件名子:音频文件名字:deepspeech转录'
    tran_txt = '/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/tran.txt' 
    audio_dir = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/new_crop"
    tran_d_txt = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/tran_d.txt"
    lm = True    # 若为True，使用语言模型，表示使用语言模型矫正无效单词
    ds_main(audio_dir, tran_txt, tran_d_txt, lm_=lm)
