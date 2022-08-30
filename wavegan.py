# coding=utf-8
import tensorflow as tf
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from audio_data_transf import tf_seq_to_frame, tf_frame_to_seq



def lrelu(inputs, alpha=0.2, name='lrelu'):
    """
     Leaky-ReLu激活函数，y = max(x*α, x)，用于解决ReLU在x<0时的硬饱和问题。
    """
    return tf.maximum(alpha*inputs, inputs, name=name)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
    b, nframe, x_len, nch = x.get_shape().as_list()  # 将x的shape转化为list，batch_size, 音频长, 声道数

    # 随机生成一个服从[minval, maxval)区间的均匀分布的数
    phase = tf.random_uniform([], minval=-rad, maxval=rad+1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)  # 取二者之间的最大值
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r

    # 填充x，由于x是3维，所以，参数的第二项中的3组值分别代表对x的三个维填充的宽度，"reflect"模式表示对原有数据的映
    # 射式的填充，而非0填充；填充的宽度值不能大于原有的维度值；在这里，填充后的输出由shape(b, x_len, nch)转化为
    # shape(b, x_len+phase, nch)
    x = tf.pad(x, [[0, 0], [0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    # 由于pad_l和pad_r总有一个为0，所以，这里只取x的第一维中包含填充的那边的x_len长度的值，x的shape仍是(b, x_len, nch)
    x = x[:, :, phase_start:phase_start+x_len]

    # set_shape只能改变tensor的形状为None的情况，且不能跨阶数改变形状，可参考：https://zhuanlan.zhihu.com/p/54130737
    x.set_shape([b, nframe, x_len, nch])
    return x


def atrous_conv1d(value, dilation, kwidth=3, num_kernels=1,
                  name='atrous_conv1d', bias_init=None, stddev=0.02):
    input_shape = value.get_shape().as_list()
    in_channels = input_shape[-1]
    assert len(input_shape) >= 3
    with tf.variable_scope(name):
        weights_init = tf.truncated_normal_initializer(stddev=0.02)
        # filter shape: [kwidth, in_channels, output_channels]
        filter_ = tf.get_variable('w', [kwidth, in_channels, num_kernels],
                                  initializer=weights_init)
        padding = [[0, 0], [(kwidth/2) * dilation, (kwidth/2) * dilation], [0, 0]]
        padded = tf.pad(value, padding, mode='SYMMETRIC')
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='SAME')
        # Remove excess elements at the end.
        result = tf.slice(restored, [0, 0, 0], [-1, input_shape[1], num_kernels])
        if bias_init is not None:
            b = tf.get_variable('b', [num_kernels],
                                initializer=tf.constant_initializer(bias_init))
            result = tf.add(result, b)
        return result


def residual_block(input_, dilation, kwidth, num_kernels=1,
                   bias_init=None, stddev=0.02, do_skip=True,
                   name='residual_block'):
    print('input shape to residual block: ', input_.get_shape())
    with tf.variable_scope(name):
        h_a = atrous_conv1d(input_, dilation, kwidth, num_kernels,
                            bias_init=bias_init, stddev=stddev)
        h = tf.tanh(h_a)
        # apply gated activation
        z_a = atrous_conv1d(input_, dilation, kwidth, num_kernels,
                            name='conv_gate', bias_init=bias_init,
                            stddev=stddev)
        z = tf.nn.sigmoid(z_a)
        print('gate shape: ', z.get_shape())
        # element-wise apply the gate
        gated_h = tf.multiply(z, h)
        print('gated h shape: ', gated_h.get_shape())
        #make res connection
        h_ = conv1d(gated_h, kwidth=1, num_kernels=1,
                    init=tf.truncated_normal_initializer(stddev=stddev),
                    name='residual_conv1')
        res = h_ + input_
        print('residual result: ', res.get_shape())
        if do_skip:
            #make skip connection
            skip = conv1d(gated_h, kwidth=1, num_kernels=1,
                          init=tf.truncated_normal_initializer(stddev=stddev),
                          name='skip_conv1')
            return res, skip
        else:
            return res


def ResBlock(x, train, filters=32, kernel_size=25, strides=1):
    """该残差块包含两个1×3一维卷积"""
    conv1 = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=(1, kernel_size), strides=(1, strides),
                             padding="same")
    conv1_norm = tf.layers.batch_normalization(conv1, training=train)
    conv1_l = tf.nn.leaky_relu(features=conv1_norm, alpha=0.2)

    conv2 = tf.layers.conv2d(inputs=conv1_l, filters=filters, kernel_size=(1, kernel_size), strides=(1, strides),
                             padding="same")
    # conv2_norm = tf.layers.batch_normalization(conv2, training=train)
    conv2_norm = tf.contrib.layers.instance_norm(conv2)
    return x + conv2_norm


def ConvNormLRelu(x, train, filters, kernel_size=3, strides=1):
    """一维卷积，conv1d ==> batch norm ==> lrelu"""
    Conv = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=(1, kernel_size), strides=(1, strides),
                            padding="same")

    # Norm = tf.layers.batch_normalization(Conv, training=train)
    Norm = tf.contrib.layers.instance_norm(Conv)
    return tf.nn.leaky_relu(Norm, alpha=0.2)


def TransConvNormRelu(x, train, filters, kernel_size=3, strides=2, upsample="zeros"):
    """一维逆卷积：tran conv1d ==> batch norm ==> relu"""
    TransConv = tf.layers.conv2d_transpose(inputs=x, filters=filters, kernel_size=(1, kernel_size),
                                           strides=(1, strides), padding="same")

    # Norm = tf.layers.batch_normalization(TransConv, training=train)
    Norm = tf.contrib.layers.instance_norm(TransConv)
    return tf.nn.leaky_relu(Norm, alpha=0.2)


"""
  Input: [batch_size, 帧数, 帧长, 1]
  Output: [batch_size, 帧数, 帧长, 1]
"""


def WaveGANGenerator(
        x,
        slice_len=16384,
        nch=1,  # nch在该代码中应该指音频的声道数，默认为1
        kernel_len=25,
        dim=64,
        use_batchnorm=False,
        upsample='zeros',
        train=True,
        phaseshuffle_rad=0):
    """生成器结构实现"""

    # assert slice_len in [16384, 32768, 65536]

    # 先对其进行几层卷积操作
    # 每层的形状变化规则：shape(bs, a, b, c) ===> shape(bs, a, 'b/strides'取上整, filters)
    # [16384, 1] -- [4096, 64] -- [1024, 128] -- [256, 256] -- [64, 512] -- [16, 1024]
    with tf.variable_scope("downConv_1-5"):
        down_conv1 = ConvNormLRelu(x, train, filters=dim, kernel_size=kernel_len, strides=4)
        down_conv2 = ConvNormLRelu(down_conv1, train, filters=dim*2, kernel_size=kernel_len, strides=4)
        down_conv2 = tf.layers.dropout(down_conv2, rate=0.3)
        down_conv3 = ConvNormLRelu(down_conv2, train, filters=dim*4, kernel_size=kernel_len, strides=4)
        down_conv4 = ConvNormLRelu(down_conv3, train, filters=dim*8, kernel_size=kernel_len, strides=4)
        down_conv5 = ConvNormLRelu(down_conv4, train, filters=dim*16, kernel_size=kernel_len, strides=4)

    # # 再进行4层残差块操作
    # # 残差块操作的输出输入shape是相同的
    # # [16, 1024] --> [16, 1024]
    # with tf.variable_scope("resBlock_6-9"):
    #     res_block1 = ResBlock(down_conv5, train, filters=dim*16)
    #     res_block2 = ResBlock(res_block1, train, filters=dim*16)
    #     res_block3 = ResBlock(res_block2, train, filters=dim*16)
    #     res_block4 = ResBlock(res_block3, train, filters=dim*16)  # shape(bs, 16, 1024)

    # 然后执行逆（或转置）卷积操作，进行上采样
    # [16, 1024] -- [64, 512] -- [256, 256] -- [1024, 128] -- [4096, 64]
    with tf.variable_scope("upconv_10-13"):
        up_conv1 = TransConvNormRelu(down_conv5, train, filters=dim*8, kernel_size=kernel_len, strides=4)
        up_conv2 = TransConvNormRelu(up_conv1, train, filters=dim*4, kernel_size=kernel_len, strides=4)
        up_conv2 = tf.layers.dropout(up_conv2, rate=0.3)
        up_conv3 = TransConvNormRelu(up_conv2, train, filters=dim*2, kernel_size=kernel_len, strides=4)
        up_conv4 = TransConvNormRelu(up_conv3, train, filters=dim*1, kernel_size=kernel_len, strides=4)

    # 最后，再进行一次逆卷积操作，不过后面没跟BN操作，激活函数是tanh
    # [4096, 64] -- [16384, 1]
    with tf.variable_scope("upConv_output"):
        output = tf.layers.conv2d_transpose(up_conv4, filters=1, kernel_size=(1, kernel_len), strides=(1, 4),
                                            padding='same')

    # 双曲正切函数，值域在(-1, 1)
    output = tf.nn.tanh(output)  # shape(bs, 帧数, 帧长, 1)，等于最初始的输入的shape

    # Automatically update batchnorm moving averages every time G is used during training
    # 若是处于训练期间，batchnorm批归一化的均值和方差更新操作定义
    # 在训练的时候，若是网络中用了batch_norm操作，需要在batch_norm操作后执行下面的操作
    if train:
        # 关于batchnorm操作以及下面这几步操作的详细介绍可参考：https://blog.csdn.net/huitailangyz/article/details/85015611
        # 返回scope所指的所有操作ops组成的list
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)

        # 该函数保证其辖域中的操作必须要在该函数所传递的参数中的操作完成后再进行
        # （即进行下面这条操作时，需要先执行update_ops中的操作）
        with tf.control_dependencies(update_ops):
            # tf.identity是返回一个与output一模一样新的tensor的op，这会增加一个新节点到gragh中，这时control_dependencies就
            # 会生效。具体参考：https://blog.csdn.net/hu_guan_jie/article/details/78495297
            output = tf.identity(output)
    return output  # 生成器的最终输出是一个shape为(bs, 帧数, 帧长, 1)，nch默认是1；value均介于-1到1之间


"""
  Input: [batch_size, 音频长, 1]
  Output: [batch_size,] (linear output)
"""
"""
有人认为，去除池化层，对生成式模型很重要，比如gan，可通过调节卷积或反卷积的stride达到下采样或上采样的效果;
因为使用的是WGAN-GP优化策略，为了防止gp梯度惩罚项和BN的冲突，所以GAN中不能使用BN，可用IN等其他归一化方法替代。
"""


def WaveGANDiscriminator(
        x,
        kernel_len=25,
        dim=64,
        use_batchnorm=False,
        phaseshuffle_rad=1):
    """判别器的结构"""

    # 输入x的初始shape(batch_size, 音频长, 1)，由于batch_size的初始shape是None，所以需要通过tf.shape(x)[0]获取其大小
    batch_size = tf.shape(x)[0]

    if use_batchnorm:  # 决定是否应用批归一化
        # norm = lambda xx: tf.layers.batch_normalization(xx, training=True)
        norm = lambda xx: tf.contrib.layers.instance_norm(xx)
    else:
        norm = lambda xx: xx

    if phaseshuffle_rad > 0:  # 决定是否phaseshuffle操作
        phaseshuffle = lambda xx: apply_phaseshuffle(xx, phaseshuffle_rad)
    else:
        phaseshuffle = lambda xx: xx

    # bs, 帧数, 16384, 1
    # Layer 0
    # [音频长, 1] -> [音频长/4, 64]
    output = x
    # 相比生成器的每层执行转置卷积和relu激活函数，判别器执行的是卷积操作和lrelu操作，判别器的第一维不断减少，最后一维在不断增加
    with tf.variable_scope('downconv_0'):
        # 一维卷积操作，第二项参数决定输出的最后一维大小，第三项参数表示核size，第4项表示步长，最后一项表示填充；由于是填充的，
        # 且步长是4，所以，输出的shape的第一维是输入时的1/4
        output = tf.layers.conv2d(output, filters=dim, kernel_size=(1, kernel_len), strides=(1, 4), padding="same")
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 1
    # [音频长/4, 64] -> [音频长/16, 128]
    with tf.variable_scope('downconv_1'):
        output = tf.layers.conv2d(output, dim*2, (1, kernel_len), (1, 4), padding='SAME')
        output = norm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 2
    # [音频长/16, 128] -> [音频长/64, 256]
    with tf.variable_scope('downconv_2'):
        output = tf.layers.conv2d(output, dim*4, (1, kernel_len), (1, 4), padding='SAME')
        output = norm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    output = tf.layers.dropout(output, rate=0.3)

    # Layer 3
    # [音频长/64, 256] -> [音频长/256, 512]
    with tf.variable_scope('downconv_3'):
        output = tf.layers.conv2d(output, dim*8, (1, kernel_len), (1, 4), padding='SAME')
        output = norm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 4
    # [音频长/256, 512] -> [音频长/1024, 1024]
    with tf.variable_scope('downconv_4'):
        output = tf.layers.conv2d(output, dim*16, (1, kernel_len), (1, 4), padding='SAME')
        output = norm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    output = tf.layers.dropout(output, rate=0.3)

    # Layer 5
    # [音频长/1024, 1024] -> [音频长/4096, 2048]
    with tf.variable_scope('downconv_5'):
        output = tf.layers.conv2d(output, dim*32, (1, kernel_len), (1, 4), padding='SAME')
        output = norm(output)
    output = lrelu(output)

    # Flatten，展平操作，将其shape变为(batchsize, ?)
    output = tf.reshape(output, [batch_size, -1])

    # Connect to single logit
    with tf.variable_scope('output'):
        # 全连接层，输出shape(batch_size, 1)，加了[:, 0]后的输出shape(batch_size,)，意味着每一样本对应的输出仅是一个数值
        output = tf.layers.dense(output, 1)[:, 0]

    # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator
    # for training
    return output





# def WaveGANDiscriminator(
#         x,
#         kernel_len=25,
#         dim=64,
#         use_batchnorm=False,
#         phaseshuffle_rad=1):
#     """判别器的结构"""
#
#     # 输入x的初始shape(batch_size, 音频长, 1)，由于batch_size的初始shape是None，所以需要通过tf.shape(x)[0]获取其大小
#     batch_size = tf.shape(x)[0]
#
#     if use_batchnorm:  # 决定是否应用批归一化
#         batchnorm = lambda xx: tf.layers.batch_normalization(xx, training=True)
#     else:
#         batchnorm = lambda xx: xx
#
#     if phaseshuffle_rad > 0:  # 决定是否phaseshuffle操作
#         phaseshuffle = lambda xx: apply_phaseshuffle(xx, phaseshuffle_rad)
#     else:
#         phaseshuffle = lambda xx: xx
#
#     # bs, 帧数, 16384, 1
#     # Layer 0
#     # [音频长, 1] -> [音频长/4, 64]
#     output = x
#     # 相比生成器的每层执行转置卷积和relu激活函数，判别器执行的是卷积操作和lrelu操作，判别器的第一维不断减少，最后一维在不断增加
#     with tf.variable_scope('downconv_0'):
#         # 一维卷积操作，第二项参数决定输出的最后一维大小，第三项参数表示核size，第4项表示步长，最后一项表示填充；由于是填充的，
#         # 且步长是4，所以，输出的shape的第一维是输入时的1/4
#         output = tf.layers.conv1d(inputs=output, filters=dim, kernel_size=kernel_len, strides=4, padding='SAME')
#     output = lrelu(output)
#     output = phaseshuffle(output)
#
#     # Layer 1
#     # [音频长/4, 64] -> [音频长/16, 128]
#     with tf.variable_scope('downconv_1'):
#         output = tf.layers.conv1d(output, dim*2, kernel_len, 4, padding='SAME')
#         output = batchnorm(output)
#     output = lrelu(output)
#     output = phaseshuffle(output)
#
#     # Layer 2
#     # [音频长/16, 128] -> [音频长/64, 256]
#     with tf.variable_scope('downconv_2'):
#         output = tf.layers.conv1d(output, dim*4, kernel_len, 4, padding='SAME')
#         output = batchnorm(output)
#     output = lrelu(output)
#     output = phaseshuffle(output)
#
#     # Layer 3
#     # [音频长/64, 256] -> [音频长/256, 512]
#     with tf.variable_scope('downconv_3'):
#         output = tf.layers.conv1d(output, dim*8, kernel_len, 4, padding='SAME')
#         output = batchnorm(output)
#     output = lrelu(output)
#     output = phaseshuffle(output)
#
#     # Layer 4
#     # [音频长/256, 512] -> [音频长/1024, 1024]
#     with tf.variable_scope('downconv_4'):
#         output = tf.layers.conv1d(output, dim*16, kernel_len, 4, padding='SAME')
#         output = batchnorm(output)
#     output = lrelu(output)
#     output = phaseshuffle(output)
#
#     # Layer 5
#     # [音频长/1024, 1024] -> [音频长/4096, 2048]
#     with tf.variable_scope('downconv_5'):
#         output = tf.layers.conv1d(output, dim*32, kernel_len, 4, padding='SAME')
#         output = batchnorm(output)
#     output = lrelu(output)
#
#     print("print:", output.shape)
#     # Flatten，展平操作，将其shape变为(batchsize, ?)
#     output = tf.reshape(output, [batch_size, -1])
#
#     # Connect to single logit
#     with tf.variable_scope('output'):
#         # 全连接层，输出shape(batch_size, 1)，加了[:, 0]后的输出shape(batch_size,)，意味着每一样本对应的输出仅是一个数值
#         output = tf.layers.dense(output, 1)[:, 0]
#
#     # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator
#     # for training
#     return output


if __name__ == "__main__":
    """仅供测试代码是否可运行"""
    # 随机生成一个shape为（1, 16384, 1）的tensor，然后分帧输入生成器，将其输出还原，输入判别器，用于检查代码是否存在问题
    a = tf.constant(value=np.arange(1, 50001) / 50001, dtype=tf.float32, shape=[1, 50000, 1])
    a_f = tf_seq_to_frame(y=a, frame_length=16384)
    print(a.shape)
    print(a_f.shape)
    g = WaveGANGenerator(a_f, train=True)
    print(g.shape)
    # g_s = tf_frame_to_seq(y_frames=g, org_len=a.shape[1])
    # print(g_s.shape)
    d = WaveGANDiscriminator(g)
    print(d.shape)
    #
    # # 以下的配置config必须加上，
    # config = tf.ConfigProto(allow_soft_placement=True)  # 用于创建会话时的参数配置
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 设置占用GPU80%的显存
    # config.gpu_options.allow_growth = True  # 开始分配少量显存，然后按需增加
    # with tf.Session(config=config) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(a_f).shape)
    #     print(sess.run(g).shape)
    #     print(sess.run(g_s).shape)
    #     print(sess.run(d).shape)
