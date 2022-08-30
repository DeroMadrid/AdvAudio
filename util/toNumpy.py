# 将tensor转化为numpy
# 方便进行音频文件的写
# 2021/8/30 by xiaoyan
import tensorflow as tf
def tensortoNumpy(ax):
    sess = tf.Session()
    with sess.as_default():
        ax_numpy = ax.eval()
    return ax_numpy