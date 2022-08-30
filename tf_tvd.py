import tensorflow as tf
from scipy.io.wavfile import read, write
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def tvd(org, adv):
    """
    通过全变分去噪（TVD）方法去除音频中的大多数脉冲，使得音频挺起来更加不可感知
    :param org: 原始音频， shape(batch_size, 音频长, 1)
    :param adv: 对抗音频， shape(batch_size, 音频长, 1)
    :return: l_tvd，标量
    """
    gama = 10   # 平衡参数
    delta = tf.subtract(adv, org)
    l_delta = tf.reduce_mean(tf.square(delta), reduction_indices=[1, 2])
    l_adv = tf.reduce_mean(tf.abs(adv[:, 2:, :] - adv[:, 1: -1, :]), reduction_indices=[1, 2])
    l_tvd = l_delta + gama * l_adv
    return tf.reduce_mean(l_tvd)


if __name__ == "__main__":
    sr1, adv = read("./adv.wav")
    sr2, org = read("./org.wav")
    org = tf.constant(org, shape=(1, org.shape[0], 1), dtype=tf.float32)
    adv = tf.constant(adv, shape=(1, adv.shape[0], 1), dtype=tf.float32)
    l = tvd(org, adv)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(org.shape)
        print(adv.shape)
        print(sess.run(l))
