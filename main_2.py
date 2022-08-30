# coding=utf-8
from functools import reduce
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 2080ti有两个GPU，这里指定0，指仅使用0号gpu，写1指1号GPU，不写或者“0,1”都写指两个GPU都使用
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import time
import pickle
import sys
import numpy as np
from glob import glob
import tensorflow as tf
from six.moves import xrange
from scipy.io.wavfile import read, write
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
import pandas
from typing import List

import loader0_new2 as loader
from wavegan import WaveGANGenerator, WaveGANDiscriminator
# from wavegan_0 import WaveGANGenerator, WaveGANDiscriminator
from audio_data_transf import tf_frame_to_seq, tf_seq_to_frame
from tf_tvd import tvd
from metric import str_list_cmp, snr

# deepspeech
import DeepSpeech
from tf_logits import get_logits
from share import toks, ds_preds, ds_preds_lm, ds_var

# lingvo
from lingvo import model_imports
from lingvo import model_registry
from lingvo.core import cluster_factory
from tool import create_features, create_inputs


sys.path.append("DeepSpeech")


def train(args, train_dir_, summary_dir_, model_dir_, train_tran_txt_, result_dir_):  # args是参数集

    """
    尽可能在跑代码之前，先将所有音频文件统一长度，所有转录也统一长度。
    """
    handle = tf.placeholder(tf.string, shape=[])  # 用于控制验证集和训练集的切换
    with tf.name_scope("loader"):
        # iterator, training_iterator, validation_iterator = loader.get_iterator(
        #     handle, args.train_batch_size, args.validation_batch_size, args.training_wav_dir,
        #     args.training_tran_txt, args.validation_wav_dir, args.validation_tran_txt)
        iterator, training_iterator = loader.get_iterator(
            handle, args.train_batch_size, args.training_wav_dir, args.training_tran_txt)

    next_element = iterator.get_next()
    x = next_element[0]  # 原始音频数据值，shape(batch_size, 音频长, 1)
    ds_org_tran_seq = next_element[1]  # deepspeech原转录的数字序列形式， shape(batch_size, org_tran_maxlen)
    ds_org_trans = next_element[2]  # deepspeech原转录，shape(batch_size,)
    lv_org_trans = next_element[3]  # lingvo原转录，shape(batch_size,)
    js_org_tran_seq = next_element[4]  # jasper原转录的数字序列形式， shape(batch_size, org_tran_maxlen)
    js_org_trans = next_element[5]  # jasper原转录
    # wl_org_trans = next_element[5]  # wav2letter+原转录
    tf_masks = next_element[6]
    tf_masks_freq = next_element[7]
    org_trans = next_element[8]
    ds_tran_maxlen = tf.shape(ds_org_tran_seq)[1]  # deepspeech原转录的最大长度
    js_tran_maxlen = tf.shape(js_org_tran_seq)[1]  # deepspeech原转录的最大长度

    # 可能需要修改x的形状去适应以前版本的gan
    x_file = tf_seq_to_frame(y=x, frame_length=16384)
    # Make generator，原音频经过G输出扰动G_x，它的shape(batch_size,音频长, 1)
    with tf.variable_scope('G'):
        G_x = WaveGANGenerator(x_file, train=True, **args.wavegan_g_kwargs)  # 参数前带双'*'，表示它是一个词典dict
        # G_x = WaveGANGenerator(x, train=True, **args.wavegan_g_kwargs)
        if args.wavegan_genr_pp:
            # 若是该参数为True，对G_x再进行一次一维卷积操作，‘用于DCGAN的后处理’，输出shape(batch_size,slice_len,1)，官方解释若是结果噪音太大，可以加这一操作
            with tf.variable_scope('pp_filt'):
                G_x = tf.layers.conv2d(G_x, 1, (1, args.wavegan_genr_pp_len), use_bias=False, padding='same')

    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')  # 将scope为‘G’的所有可训练的变量赋给G_var列表

    # Print G summary；打印出G所用的总参数量
    print('-' * 80)
    print('Generator vars')
    nparams = 0
    for v in G_vars:  # 遍历G_vars中所有变量
        v_shape = v.get_shape().as_list()  # 返回每一个变量的shape
        v_n = reduce(lambda x1, y1: x1 * y1, v_shape)  # 将shape中的值相乘，shape中的各个值相乘的结果是该变量中包含的参数量
        nparams += v_n  # 将所有变量的参数量累计相加
    #     print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))  # v.name表示变量的名字
    print(
        'print G ~ Total params: {} ({:.2f} MB)'.format(nparams,
                                                        (float(nparams) * 4) / (1024 * 1024)))  # 打印G的参数量，单位MB

    #########################
    thresh = 0.062
    noise_f_raw = tf.where(condition=tf.equal(x_file, 0), x=x_file, y=tf.clip_by_value(G_x, -thresh, thresh))
    ax_f = tf.clip_by_value(x_file + noise_f_raw, -1, 1)  # shape(batch_size,帧数, 帧长, 1)
    ## noise_raw = tf.clip_by_value(G_x, -thresh, thresh) * tf.expand_dims(tf_masks, axis=-1)
    ##  ax = tf.clip_by_value(x + noise_raw, -1, 1)
    ax = tf_frame_to_seq(y_frames=ax_f, org_len=x.shape[1])
    disturb = ax - x  # 扰动
    # thresh = 0.05
    # noise_raw = tf.clip_by_value(G_x, -thresh, thresh) * tf.expand_dims(tf_masks, axis=-1)
    # ax = tf.clip_by_value(x + noise_raw, -1, 1)
    # disturb = ax - x  # 扰动
    ########################
    # 关于summary以及tensorboard可视化的使用，可参考：https://www.jianshu.com/p/b75895dc231e
    # 或 https://blog.csdn.net/gsww404/article/details/78605784
    # Summarize；summary对象可以用tensorboard显示；下面每一项的参数：(name，tensor)，audio项还有一个采样率参数；
    # audio展示训练过程中记录的音频;  scalar用来显示标量信息;  histogram用来显示直方图信息；
    # 注意，传入的音频数据x必须在[-1,1]之间；shape可以是(bs,frames,nch)或(bs,frames)。
    # tf.summary.audio(name='x', tensor=x, sample_rate=args.data_sample_rate)
    # tf.summary.audio('G_x_s', G_x_s, args.data_sample_rate)
    # tf.summary.audio('adv_x', ax, args.data_sample_rate)  # 分别是原始、扰动、对抗音频
    # G_x_rms = tf.sqrt(tf.reduce_mean(tf.square(G_x_s[:, :, 0]), axis=1))
    # x_rms = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, 0]), axis=1))
    # ax_rms = tf.sqrt(tf.reduce_mean(tf.square(ax[:, :, 0]), axis=1))  # 每一样本对应的扰动、原音频、对抗音频的L2范数，[bs,1]
    # tf.summary.histogram(name='x_rms_batch', values=x_rms)  # 表示以直方图形式显示原音频的L2范数
    # tf.summary.histogram('G_x_rms_batch', G_x_rms)
    # tf.summary.histogram('ax_rms_batch', ax_rms)
    # tf.summary.scalar(name='x_rms', tensor=tf.reduce_mean(x_rms))
    # tf.summary.scalar('G_x_rms', tf.reduce_mean(G_x_rms))
    # tf.summary.scalar('ax_rms', tf.reduce_mean(ax_rms))  # 该batch中每一样本对应的原、扰动、对抗音频的L2范数均值

    # Make real discriminator，将原（真）音频数据 x 输入判别器，得到其输出 D_x，其shape(batch_size,)
    with tf.name_scope('D_x'), tf.variable_scope('D'):
        D_x = WaveGANDiscriminator(x_file, **args.wavegan_d_kwargs)
        # D_x = WaveGANDiscriminator(x, **args.wavegan_d_kwargs)
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')  # 下面执行与生面的生成器后面差不多的几步操作
    print(D_vars)

    # Print D summary；打印出D所用的总参数量
    print('-' * 80)
    print('Discriminator vars')
    nparams = 0
    for v in D_vars:
        v_shape = v.get_shape().as_list()
        v_n = reduce(lambda x1, y1: x1 * y1, v_shape)
        nparams += v_n
        # print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
    print('print D ~ Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
    print('-' * 80)

    # Make fake discriminator，将对抗（假）音频数据 ax 输入判别器，得到其输出 D_ax，其shape(batch_size,)
    with tf.name_scope('D_G_ax'), tf.variable_scope('D', reuse=True):
        D_ax = WaveGANDiscriminator(ax_f, **args.wavegan_d_kwargs)
        # D_ax = WaveGANDiscriminator(ax, **args.wavegan_d_kwargs)

    # Create loss，以下是几种损失函数的选择
    D_clip_weights = None
    if args.wavegan_loss == 'dcgan':  # 'dcgan'计算loss使用的是交叉熵损失函数
        fake = tf.zeros([args.train_batch_size], dtype=tf.float32)  # 假样本的标签，与判别器输出同shape，值全为0
        real = tf.ones([args.train_batch_size], dtype=tf.float32)  # 真样本的标签，与判别器输出同shape，值全为1
        fake = tf.add(fake, 0.1)
        real = tf.subtract(real, 0.1)  # 起到平滑标签的作用，避免判别器过度自信
        # 生成器G的loss，它的目的是使得判别器D将其生成的对抗样本ax判别为真样本，所以标签是1
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_ax, labels=real))
        # 判别器D的损失loss1，它的目的是将ax判别为假样本，所以标签是0
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_ax, labels=fake))
        # 判别器D的loss2，它的目的是将原音频 x 判别为真，所以标签是1
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x, labels=real))

        D_loss /= 2.  # 然后将判别器的两个损失loss相加，再除以2，作为判别器的最终的损失

    elif args.wavegan_loss == 'lsgan':  # 'lsgan'计算loss的方式是：(logits - label)**2，即最小二乘法
        G_loss = tf.reduce_mean((D_ax - 1.) ** 2)  # 标签是啥，减号“-”后面就是啥，与dcgan对应
        D_loss = tf.reduce_mean((D_x - 1.) ** 2)
        D_loss += tf.reduce_mean(D_ax ** 2)
        D_loss /= 2.

    elif args.wavegan_loss == 'wgan':
        # 'wgan'计算loss的思想是：（G的目标是最大化‘D对于对抗音频’的输出； D目标是最大化‘原音频’输出，最小化‘对抗音频’输出）
        # 并且将判别器的所有参数变量值裁剪到[-.01, .01]范围内，即梯度裁剪。
        G_loss = -tf.reduce_mean(D_ax)  # 最大化D_ax的均值，相当于最小化该值的负数
        D_loss = tf.reduce_mean(D_ax) - tf.reduce_mean(D_x)

        with tf.name_scope('D_clip_weights'):
            clip_ops = []
            for var in D_vars:
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            D_clip_weights = tf.group(*clip_ops)  # tf.group用于将多个操作ops组成一组，一起执行

    elif args.wavegan_loss == 'wgan-gp':
        # 'wgan-gp'将梯度裁剪改为梯度惩罚;
        # 该方法要求判别器D不能使用BN（会引入同一batch样本的相互依赖关系），可以使用其他的，如LN、IN
        G_loss = -tf.reduce_mean(D_ax)
        D_loss = tf.reduce_mean(D_ax) - tf.reduce_mean(D_x)

        alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1, 1], minval=0., maxval=1.)
        # alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
        ## interpolates = alpha*x + (1-alpha)*ax
        interpolates = x_file + alpha * (ax_f - x_file)  # 计算原样本和生成的对抗样本间的插值
        # interpolates = x + alpha * (ax - x)  # 计算原样本和生成的对抗样本间的插值
        with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):  # with辖域的变量名会以“D_interp/D/”开头
            D_interp = WaveGANDiscriminator(interpolates, **args.wavegan_d_kwargs)

        LAMBDA = 10
        # 若是tf.gradients中给stop_gradients赋了值，那么在求梯度时，只能求到它这儿，不能再往前传播求梯度;
        # tf.gradients是计算参数1（中各个值）对参数2（中各个变量）的梯度，
        # 并将对同一变量求的所有梯度相加，即它的返回值的shape同参数2中的变量的shape;
        gradients = tf.gradients(D_interp, [interpolates])[0]  # 计算D_interp关于插值的梯度
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))  # 计算该batch中每一样本的插值梯度的L2范数
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
        D_loss += LAMBDA * gradient_penalty
    else:
        raise NotImplementedError()

    # ——————————————————————————————————分界线——————————————————————————————————————————————————————————————————————————
    # 计算对抗损失adv_loss，目的是让对抗音频经ASR的输出与原音频转录尽可能差得远（或与目标转录尽可能差的少）
    # 将对抗音频输入target-ASR ------------------------------------------------------------------------------
    adv_input = ax[:, :, 0]  # shape(bs, 音频长度)
    fs = args.data_sample_rate  # 采样率
    random_noise = tf.random_normal(tf.shape(ax[:, :, 0]), stddev=2)
    batch_size = args.train_batch_size

    # 计算全变分去噪作用的loss，用于降低对抗音频的噪声
    tvd_loss = tvd(x, ax)

    # 生成噪声的铰链损失
    zeros = tf.zeros((tf.shape(x)[0]))
    l2_loss = tf.reduce_mean(tf.maximum(zeros, tf.norm(tf.reshape(disturb, (tf.shape(x)[0], -1)), axis=1) - thresh))
    # l2_loss = tf.norm((ax - x), ord=2, axis=None, keep_dims=False, name="l2loss")
    # ~~~~~~~~~~~~~~~~~~~~~~~deepspeech~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 将adv_input输入deepspeech-0.4模型，输出其logits，作为计算ctc-loss的inputs
    adv_input_int16 = adv_input * 32768 + random_noise
    deepspeech_pass_in = tf.clip_by_value(t=adv_input_int16, clip_value_min=-2 ** 15, clip_value_max=2 ** 15 - 1)
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        deepspeech_logits, seq_len = get_logits(new_input=deepspeech_pass_in)  # 调用deepspeech，返回logits值，用于计算ctc-loss
    ds_decodes, _ = tf.nn.ctc_beam_search_decoder(deepspeech_logits, seq_len, merge_repeated=False, beam_width=100)
    deepspeech_logits_lm = tf.transpose(tf.nn.softmax(deepspeech_logits), [1, 0, 2])  # 带语言模型解码时的输入
    # ---------------------------end------------------------------------------------------------------------------------

    if not args.targeted:
        # 无目标（首先将原音频转录的数字序列形式 转化为 稀疏Tensor，作为计算ctc-loss的labels）-----------------------------
        ds_org_tran_seq = tf.clip_by_value(ds_org_tran_seq, 0, len(toks()) - 2)
        org_tran_len = tf.fill(dims=(args.train_batch_size,), value=tf.cast(ds_tran_maxlen, tf.int32))
        org_tran_sparse = ctc_label_dense_to_sparse(labels=ds_org_tran_seq, label_lengths=org_tran_len)
        # 计算ctc-loss，shape(batch_size,)，既然是无目标，则要让原转录和对抗音转录的loss值越大越好，所以前面加“-”
        ctcloss = tf.nn.ctc_loss(labels=tf.cast(org_tran_sparse, tf.int32), inputs=deepspeech_logits,
                                 sequence_length=seq_len)
        deepspeech_adv_loss = -tf.reduce_mean(ctcloss)  # deepspeech的对抗损失
        # deepspeech_adv_loss = tf.reciprocal(tf.reduce_mean(ctcloss))  # deepspeech的对抗损失。取倒数
    else:
        # 有目标（首先将指定的目标短语转化为稀疏Tensor，然后计算-------------------------------------------------------------
        tgt_tran = args.adv_target  # 自定义的目标转录
        tgt_tran_seq_list = [[toks().index(j) for j in tgt_tran] for _ in
                             range(args.train_batch_size)]  # 将目标转录短语转为数字序列形式
        tgt_tran_seq = tf.constant(np.array([list(t) + [0] * (len(tgt_tran) - len(t)) for t in tgt_tran_seq_list]),
                                   dtype=tf.int32)  # 目标转录
        tgt_tran_len_tf = tf.constant(np.array([len(t) for t in tgt_tran_seq_list]), dtype=tf.int32)
        tgt_tran_sparse = ctc_label_dense_to_sparse(labels=tgt_tran_seq, label_lengths=tgt_tran_len_tf)  # 转化为稀疏Tensor
        # 计算ctc-loss，shape(batch_size,)，计算的是目标转录和对抗音频的转录之间的ctc-loss
        ctcloss = tf.nn.ctc_loss(labels=tf.cast(tgt_tran_sparse, tf.int32), inputs=deepspeech_logits,
                                 sequence_length=seq_len)
        deepspeech_adv_loss = tf.reduce_mean(ctcloss)  # deepspeech的对抗损失
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ………………………………………………………………………………………lingvo………………………………………………………………………………………………………………………………………………………………………………………………………
    input_tf = tf.clip_by_value(ax[:, :, 0] * 32768 + random_noise, -2 ** 15, 2 ** 15 - 1)

    if not args.targeted:  # 如果是无目标攻击，tgt_tf是原始转录文本
        tgt_tf = lv_org_trans
    else:  # 若是有目标攻击，tgt_tf是自定义的目标转录文本
        tgt_tf = tf.constant(value=args.adv_target, shape=(args.train_batch_size,), dtype=tf.string)  # shape(bs, )

    sample_rate_tf = fs
    batch_size = args.train_batch_size

    tf.set_random_seed(1234)
    params = model_registry.GetParams('asr.librispeech.Librispeech960Wpm', 'Test')
    params.cluster.worker.gpus_per_replica = 1
    cluster = cluster_factory.Cluster(params.cluster)
    with cluster, tf.device(cluster.GetPlacer()):
        params.vn.global_vn = False
        params.random_seed = 1234
        params.is_eval = True
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            model = params.cls(params)
        task = model.GetTask()

        # generate the features and inputs
        features = create_features(input_tf, sample_rate_tf, tf_masks_freq)
        inputs = create_inputs(model, features, tgt_tf, batch_size, tf_masks_freq)

        # loss
        metrics = task.FPropDefaultTheta(inputs)
        lingvo_loss = tf.get_collection("per_loss")[0]

        # prediction
        lingvo_decoded = task.Decode(inputs)

        # lingvo的对抗损失
        if not args.targeted:  # 无目标时，计算的是原转录和对抗转录之间的损失，我们希望该损失尽可能的大，所以前面加负号
            lingvo_adv_loss = -tf.reduce_mean(lingvo_loss)
            # lingvo_adv_loss = tf.reciprocal(tf.reduce_mean(lingvo_loss))
        else:  # 有目标时，计算的是目标转录和对抗转录之间的损失，我们希望该损失尽可能的小，所以前面加不加负号
            lingvo_adv_loss = tf.reduce_mean(lingvo_loss)
    # ………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………

    # Summarize audios
    # tf.summary.audio(name='adv_input', tensor=tf.clip_by_value(adv_input, -1, 1), sample_rate=fs,
    #                  max_outputs=args.adv_max_outputs)
    tf.summary.scalar(name='deepspeech_adv_loss', tensor=deepspeech_adv_loss)
    tf.summary.scalar(name='lingvo_adv_loss', tensor=lingvo_adv_loss)
    # tf.summary.scalar(name='wav2letter_adv_loss', tensor=wav2letter_adv_loss)
    tf.summary.scalar(name='tvd_loss', tensor=tvd_loss)
    tf.summary.scalar(name='l2_loss', tensor=l2_loss)
    merged_summary = tf.summary.merge_all()
    # ——————————————————————————————————分界线———————————————————————————————————————————————————————————————————————————

    # Create (recommended) optimizer，优化器选择
    if args.wavegan_loss == 'dcgan':
        G_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
        D_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
    elif args.wavegan_loss == 'lsgan':
        G_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
        D_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
    elif args.wavegan_loss == 'wgan':
        G_opt = tf.train.RMSPropOptimizer(learning_rate=5e-5)
        D_opt = tf.train.RMSPropOptimizer(learning_rate=5e-5)
    elif args.wavegan_loss == 'wgan-gp':  # learning_rate=1e-4, beta1=0.5, beta2=0.9
        G_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
        D_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
    else:
        raise NotImplementedError()

    with tf.name_scope("Train_opt"):
        # Create training ops，分别计算G和D的loss关于var_list中的变量的梯度；
        # 并通过梯度和lr等参数更新var_list中的变量，使得loss值不断减小。 global_step是关系到学习率lr更新的一参数；
        # 关于deepspeech_adv_loss前面的参数，0.1时，噪声过大，0.0001时，200多次迭代，噪声快没了，转录与原音频几乎相等，
        # 注：各个loss之间，如果想让让某一loss起的作用大一些，那么它前面的参数就偏大些。
        # 正则化是为了防止过拟合，只对权重（不包括偏置）做正则化。reg = (w**2 / 2) * scale
        G_kernel_vars = [gkv for gkv in G_vars if 'kernel' in gkv.name]
        penalty_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=0.00003),
                                                              G_kernel_vars)

        # wt = tf.Variable(initial_value=np.array([1, 1, 1, 1]), dtype=tf.float32, name='other_wt')
        # gloss = tf.reduce_sum(wt * tf.stack([G_loss, js_adv_loss, l2_loss, penalty_loss]))
        # wt = tf.Variable(initial_value=np.array([1, 1, 20, 1, 8, 1]), dtype=tf.float32, name='other_wt')
        wt = tf.Variable(initial_value=np.array([1, 8, 5, 4, 1]), dtype=tf.float32, name='other_wt')
        # wt = tf.Variable(initial_value=np.array([1, 1.5, 21, 1.5, 3, 1]), dtype=tf.float32, name='other_wt')
        # wt = tf.Variable(initial_value=np.array([1, 1.5, 21, 1.5, 6, 1]), dtype=tf.float32, name='other_wt')
        # wt = tf.Variable(initial_value=np.array([ 1, 1, 30, 1, 1, 1]), dtype=tf.float32, name='other_wt')
        # gloss = tf.reduce_sum(wt * tf.stack([G_loss, deepspeech_adv_loss, lingvo_adv_loss, l2_loss, penalty_loss]))
        gloss = tf.reduce_sum(wt * tf.stack([G_loss, deepspeech_adv_loss, lingvo_adv_loss, tvd_loss, penalty_loss]))
        gstep = tf.train.get_or_create_global_step()

        # grads = tf.gradients(G_loss, G_vars)
        # grads = tf.gradients(js_adv_loss, G_vars)
        G_train_op = G_opt.minimize(loss=gloss, var_list=G_vars, global_step=gstep)
        D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

    # ------------------------------------------------------------------------------------------------------------------

    generator_var = [x for x in tf.global_variables() if x.name.startswith('G')]
    discriminator_var = [x for x in tf.global_variables() if x.name.startswith('D')]

    # 若是保存ckpt的话，var_list对应的是仅你想保存的变量组成的list，不想保存的变量不用加入该list
    g_saver = tf.train.Saver(var_list=generator_var)

    deepspeech_var = ds_var(tf.global_variables())
    if deepspeech_var:  # 若deepspeech_var不为空。若是加载ckpt的话，var_list对应的是所要加载的模型中包含的所有变量组成的list
        deepspeech_saver = tf.train.Saver(var_list=deepspeech_var)

    lingvo_var = [x for x in tf.global_variables() if x.name.startswith('librispeech')]
    if lingvo_var:
        lingvo_saver = tf.train.Saver(var_list=lingvo_var)

    other_var = [x for x in tf.global_variables() if x not in generator_var]
    other_saver = tf.train.Saver(var_list=other_var)

    # Run training
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)  # 用于创建会话时的参数配置
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 设置占用GPU80%的显存
    config.gpu_options.allow_growth = True  # 开始分配少量显存，然后按需增加


    with tf.Session(config=config) as sess:

        # 若要加载外部预训练模型（如，deepspeech），需要在全局变量初始化之后再加载（restore）；或者使用局部变量初始化方式。
        # 因为外部预训练模型的参数值已有，不需要初始化。
        sess.run(tf.global_variables_initializer())

        training_handle = sess.run(training_iterator.string_handle())
        # validation_handle = sess.run(validation_iterator.string_handle())

        writer = tf.summary.FileWriter(logdir=summary_dir_, graph=sess.graph)  # 指定文件夹保存summary信息

        deepspeech_saver.restore(sess, save_path=tf.train.latest_checkpoint("./deepSpeech_model/ds-0.4.1-ckpt"))

        lingvo_saver.restore(sess, save_path="./lingvo_model/ckpt-00908156")
        dec_metrics_dict = task.CreateDecoderMetrics()

        # tensorboard --logdir=summury所在文件夹；  http://localhost:6006
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        snrs = 0
        step = 0
        overlap_same = -1  # 统计 存在相同单词数 的数量  验证集
        overlap_diff = -1  # 统计 存在不同单词数 的数量
        cur_same = 0
        cur_diff = 0
        start = time.time()

        ds_lm = True  # 若为True，解码deepspeech转录时带语言模型矫正

        while step <= 20000:
            print("step/20000~~: ", step)

            # 训练判别器
            for i in xrange(args.wavegan_disc_nupdates):
                # 每次都是先更新wavegan_disc_nupdates次判别器，再更新一次生成器
                # sess.run(D_train_op)
                sess.run(D_train_op, feed_dict={handle: training_handle})

                # Enforce Lipschitz constraint for WGAN
                if D_clip_weights is not None:
                    # sess.run(D_clip_weights)
                    sess.run(D_clip_weights, feed_dict={handle: training_handle})

            snr_a_v = 0  # 记录该批数据的平均信噪比
            if step % 100 == 0:
                overlap_same_v = 0  # 记录该批数据的原转录（目标转录）与对抗转录中存在相同单词的样本数
                overlap_diff_v = 0  # 记录该批数据的原转录（目标转录）与对抗转录中存在不同单词的样本数

                # oa:原音频, aa:对抗音频，dot:deepspeech的原转录， dat：deepspeech的对抗转录，
                # lot：lingvo原转录，lat：lingvo对抗转录， slen用于deepspeech解码
                fetches = [x, ax, org_trans, ds_org_trans, deepspeech_logits_lm, ds_decodes[0],
                           seq_len, lv_org_trans, lingvo_decoded,
                           G_train_op, merged_summary, disturb, G_loss, deepspeech_adv_loss,
                           lingvo_adv_loss, l2_loss, seq_len, penalty_loss, gloss]
                val_r = sess.run(fetches, feed_dict={handle: training_handle})
                _, ms, delta, gl, ds_al, lv_al, l2l, slen, pl, all_gl = val_r[9:]

                org = [str(i, encoding='utf-8').lower() for i in val_r[2]]

                # 每**次迭代，查看转录结果
                dot = [str(i, encoding='utf-8').lower() for i in val_r[3]]  # deepspeech原音频转录
                lot = [str(i, encoding='utf-8').lower() for i in val_r[7]]  # lingvo原音频转录
                lat = [str(i, encoding='utf-8').lower() for i in val_r[8]['topk_decoded'][:, 0]]  # lingvo对抗转录

                if ds_lm:
                    dat = ds_preds_lm(val_r[4], val_r[6])  # deepspeech对抗音频转录
                else:
                    dat = ds_preds(val_r[5])

                if not args.targeted:  # 无目标（原转录vs对抗转录；期望存在相同的转录数为0， 存在不同的是batchsize）
                    lv_same, lv_diff = str_list_cmp(lot, lat)  # 分别是存在相同单词的转录数，存在不同单词的数
                    ds_same, ds_diff = str_list_cmp(dot, dat)
                    overlap_same_v += max(lv_same, ds_same)
                    overlap_diff_v += min(lv_diff, ds_diff)

                    # overlap_same_v += lv_same     # 如果只用lingvo
                    # overlap_diff_v += lv_diff

                else:  # 有目标 （目标转录vs对抗转录；期望存在相同单词的转录数量为batchsize， 存在不同的是0）
                    target_trans = [args.adv_target for _ in range(args.validation_batch_size)]
                    lv_same, lv_diff = str_list_cmp(lat, target_trans)  # 存在相同单词的转录数量，存在不同单词的数
                    ds_same, ds_diff = str_list_cmp(dat, target_trans)
                    overlap_same_v += min(lv_same, ds_same)
                    overlap_diff_v += max(lv_diff, ds_diff)

                    # overlap_same_v += lv_same     # 如果只用lingvo
                    # overlap_diff_v += lv_diff

                fp = open(train_tran_txt_, mode="a")
                for bs in range(args.train_batch_size):
                    fp.write("step-{0}-{1}-org_ds: {2}\n".format(step, bs, dot[bs]))
                    fp.write("step-{0}-{1}-adv_ds: {2}\n".format(step, bs, dat[bs]))
                    fp.write("step-{0}-{1}-org_lv: {2}\n".format(step, bs, lot[bs]))
                    fp.write("step-{0}-{1}-adv_lv: {2}\n\n".format(step, bs, lat[bs]))
                fp.close()

                # 保存该批生成的对抗音频和对应的原音频
                oa = val_r[0][:, :, 0] * 32768  # 原音频
                aa = val_r[1][:, :, 0] * 32768  # 对抗音频
                for n in range(args.train_batch_size):
                    valid_len = np.where(abs(oa[n, :]) > 0)[0][-1] + 1  # 返回该音频数组中最后一个非0数值的索引+1
                    f_name_l = ['org_audio', 'adv_audio']

                    org_wav = np.array(np.clip(np.round(oa[n, :valid_len]), -2 ** 15, 2 ** 15 - 1), dtype=np.int16)
                    adv_wav = np.array(np.clip(np.round(aa[n, :valid_len]), -2 ** 15, 2 ** 15 - 1), dtype=np.int16)
                    org_wav = np.pad(org_wav, (0, 100), 'constant')
                    adv_wav = np.pad(adv_wav, (0, 100), 'constant')  # 给其后填充长度为100的0

                    wav_snr = snr(org_wav, adv_wav)  # 信噪比
                    snr_a_v += wav_snr
                    for fn in f_name_l:
                        audio_name = str(step) + "_" + str(n) + '_' + ''.join(org[n]) + ".wav"
                        f_name = os.path.join(train_dir_, fn)
                        if not os.path.exists(f_name):
                            os.mkdir(f_name)
                        audio_data = adv_wav if 'adv' in fn else org_wav
                        write(filename=os.path.join(f_name, audio_name), rate=16000, data=audio_data)
                snrs_t = snr_a_v / args.train_batch_size
                # print("有目标攻击期望的是存在‘不同’的数为0")
                # print("无目标攻击期望的是存在‘相同’的数为0")
                print("****** 训练集当前批次平均信噪比：", snrs_t)
                print("****** 训练集，有目标:{0}, 总数:{1}, 存在相同数的数量:{2}，存在不同数的数量:{3} ".
                      format(args.targeted, args.train_batch_size, overlap_same_v, overlap_diff_v))
                print("***deepspeech loss:", ds_al, " ***lingvo loss:", lv_al)
                with open(train_tran_txt_, mode="a") as fp:
                    txt_time = time.time() - start
                    fp.write("step-{0}-{1:.5f}-{2}-{3}--{4}\n".format(step, txt_time, overlap_same_v, overlap_diff_v,
                                                                      snrs_t))

            else:
                # 训练生成器
                # ms:merged_summary，delta：扰动，gl：生成器损失值，ds_al：deepspeech对抗损失值，lv_al：lingvo损失值，
                # tl：tvd损失值，glosses：前几个损失值之和，l2：正则项
                fetchs = [G_train_op, merged_summary, disturb, G_loss, deepspeech_adv_loss,
                          lingvo_adv_loss, l2_loss, seq_len, penalty_loss, gloss]
                _, ms, delta, gl, ds_al, lv_al, l2l, slen, pl, all_gl = \
                    sess.run(fetchs, feed_dict={handle: training_handle})

            if step % 100 == 0:
                with open(os.path.join(result_dir_, "loss.txt"), 'a') as f_loss:
                    # f_loss.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".
                    #              format(gl, ds_al, lv_al, l2l, pl, np.sum(np.abs(delta)), avg_snr))
                    f_loss.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".
                                 format(gl, ds_al, lv_al, l2l, pl, np.sum(np.abs(delta)), snrs_t))
                # print('gl: {0}, ds_al: {1}, lv_al: {2}, tl: {3}, pl: {4}'.format(gl, ds_al, lv_al, tl, pl))
                # print("******扰动大小及信噪比", np.max(delta), "   ", np.min(delta), "  ", np.sum(delta), "  ", snrs, "  ", snrs_t)
                # print("****** lingvo-loss: {0} \t deepspeech-loss: {1}".format(lv_al, ds_al))

                # 保存模型和summary
            if step % 100 == 0 and ((not args.targeted and overlap_diff_v >= cur_diff) or
                                    (args.targeted and overlap_same_v >= cur_same)):
                g_saver.save(sess, os.path.join(model_dir_, 'g-model'), global_step=step)
                print("---------save model---------------------------------------------")
                writer.add_summary(ms, global_step=step)

                cur_same = overlap_same_v
                cur_diff = overlap_diff_v

                # 运行终止条件
            if snrs > 20 and ((not args.targeted and overlap_same == 0) or (args.targeted and overlap_diff == 0)):
                print("满足终止条件，停止运行！", file=sys.stderr)
                g_saver.save(sess, os.path.join(model_dir_, 'g-model'), global_step=step)
                writer.add_summary(ms, global_step=step)
                break

            step += 1


def mains():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    # 这种第一项参数前没‘--’，执行命令时，直接在后面加上其值即可，比如，python **.py train 即可;
    # 由于这种参数在命令行中只写其值，不好区分，所以一般默认是按其在代码中的先后顺序决定命令行中这种值的顺序.
    # parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
    # parser.add_argument('train_dir', type=str, help='Training directory')

    adv_args = parser.add_argument_group('Adversarial Example')
    # 指将被添加对抗的音频文件，第一个参数前带有“--”，执行命令示例：python **.py --adv_input no.wav
    adv_args.add_argument('--adv_target', type=str, help='Target command')
    adv_args.add_argument('--adv_magnitude', type=float,
                          help='Parameter specifing how much the perturbation be diminished')
    adv_args.add_argument('--adv_confidence', type=float,
                          help='Parameter specifing how strong the adversarial example should be')
    adv_args.add_argument('--adv_lambda', type=float, help='Lambda for adversarial loss')
    adv_args.add_argument('--adv_max_outputs', type=int, help='Number of adversarial examples to dump')
    adv_args.add_argument('--targeted', action='store_true', help='Target attack or no target attack')

    data_args = parser.add_argument_group('Data')
    data_args.add_argument('--training_wav_dir', type=str, help='training_set')
    data_args.add_argument('--validation_wav_dir', type=str, help='validation_set')
    data_args.add_argument('--training_tran_txt', type=str, help='org_tran for trainset')
    data_args.add_argument('--validation_tran_txt', type=str, help='org_tran for verifyset')

    # data_args.add_argument('--train_data_dir', type=str, help='Data directory containing *only* audio files to load')
    # data_args.add_argument('--verify_data_dir', type=str, help='Data directory containing *only* audio files to load')
    # data_args.add_argument('--train_tran_txt', type=str, help='org_tran for trainset')
    # data_args.add_argument('--verify_tran_txt', type=str, help='org_tran for verifyset')

    data_args.add_argument('--data_sample_rate', type=int, help='Number of audio samples per second')
    data_args.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536],
                           help='Number of audio samples per slice (maximum generation length)')
    data_args.add_argument('--data_num_channels', type=int,
                           help='Number of audio channels to generate (for >2, must match that of data)')
    data_args.add_argument('--data_normalize', action='store_true', dest='data_normalize',
                           help='If set, normalize the training examples')
    data_args.add_argument('--data_fast_wav', action='store_true', dest='data_fast_wav',
                           help='If your data is comprised of standard WAV files (16-bit signed PCM or 32-bit float), use this flag to decode audio using scipy (faster) instead of librosa')
    data_args.add_argument('--data_prefetch_gpu_num', type=int,
                           help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')  # 如果非负，表示即将使用的gpu号

    wavegan_args = parser.add_argument_group('WaveGAN')
    wavegan_args.add_argument('--wavegan_latent_dim', type=int,
                              help='Number of dimensions of the latent space')
    wavegan_args.add_argument('--wavegan_kernel_len', type=int,
                              help='Length of 1D filter kernels')  # 指wavgan.py中的kernel_len参数值
    wavegan_args.add_argument('--wavegan_dim', type=int,
                              help='Dimensionality multiplier for model of G and D')  # 指wavgan.py中的dim参数值
    wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',
                              help='Enable batchnorm')
    wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,
                              help='Number of discriminator updates per generator update')  # 指更新一次生成器，更新多少次判别器
    wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
                              help='Which GAN loss to use')
    wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn'],
                              help='Generator upsample strategy')
    wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', dest='wavegan_genr_pp',
                              help='If set, use post-processing filter')
    wavegan_args.add_argument('--wavegan_genr_pp_len', type=int, help='Length of post-processing filter for DCGAN')
    wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int, help='Radius of phase shuffle operation')

    train_args = parser.add_argument_group('Train')
    train_args.add_argument('--train_batch_size', type=int, help='Batch size')
    train_args.add_argument('--train_save_secs', type=int, help='How often to save model')
    train_args.add_argument('--train_summary_secs', type=int, help='How often to report summaries')

    train_args = parser.add_argument_group('validation')
    train_args.add_argument('--validation_batch_size', type=int, help='Batch size of vaidation set')

    model_args = parser.add_argument_group('Model')
    model_args.add_argument('--model_path', type=str, help='G model path for test or finetune')
    model_args.add_argument('--other_model_path', type=str, help='other model path for test')

    # 为参数设置初始默认值，运行模型时，若不在命令行输入相关参数值，就在这里直接赋值好
    parser.set_defaults(
        adv_target='custom target phrase'.lower().strip(),  # 自定义的目标转录文本
        # adv_target='zero zero zero'.lower().strip(),  # 自定义的目标转录文本
        adv_confidence=0.05,
        adv_magnitude=0.2,
        adv_lambda=0.1,
        adv_max_outputs=32,
        targeted=False,

        # training_wav_dir="./data/recaptchaV2/recaptcha5k/train",  # 训练集音频文件目录
        # training_wav_dir="/media/ps/data/gxy/Adv_audio/data/recaptchaV2/val/wav",  # 训练集音频文件目录
        # training_wav_dir="/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/new_crop",  # 训练集音频文件目录
        training_wav_dir="/media/ps/data/gxy/Adv_audio/data/digit/uniform",
        # validation_wav_dir="./data/recaptchaV2/recaptcha5k/val",  # 验证集音频文件所在目录
        # training_tran_txt='./data/recaptchaV2/recaptcha5k/trantrain_d_l_j_w_p.txt',  # 训练集转录文件（测试时，只需将验证的路径改为测试集的）
        # training_tran_txt='/media/ps/data/gxy/Adv_audio/data/recaptchaV2/val/tran_d_l_j_p.txt',  # 训练集转录文件（测试时，只需将验证的路径改为测试集的）
        # training_tran_txt='/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/tran_d_l_j_p.txt',
        # 训练集转录文件（测试时，只需将验证的路径改为测试集的）
        training_tran_txt='/media/ps/data/gxy/Adv_audio/data/digit/tran_d_l_j_p.txt',  # 训练集转录文件（测试时，只需将验证的路径改为测试集的）
        # validation_tran_txt='./data/recaptchaV2/recaptcha5k/tranval_d_l_j_w_p.txt',  # 验证集转录文件

        data_sample_rate=16000,
        data_slice_len=16384,
        data_num_channels=1,
        data_normalize=False,
        data_fast_wav=True,
        data_prefetch_gpu_num=0,

        wavegan_latent_dim=100,
        # 表示作为生成器输入的 隐向量(latent vector，我按字面意思翻译成‘隐向量’，隐向量表示服从某一分布的数组) 的第一维的size，隐向量是二维，第0维是batch_size
        # wavegan_kernel_len=28,
        wavegan_kernel_len=25,
        wavegan_dim=64,
        wavegan_batchnorm=True,
        wavegan_disc_nupdates=1,
        wavegan_loss='wgan-gp',
        wavegan_genr_upsample='zeros',
        wavegan_genr_pp=False,
        wavegan_genr_pp_len=512,
        wavegan_disc_phaseshuffle=2,

        train_batch_size=8,
        train_save_secs=300,
        train_summary_secs=120,

        validation_batch_size=16,
        model_path="./result/recaptcha/03-24-21-56_wgan-gptarget_libri/model",
        other_model_path="./result/recaptcha/03-24-21-56_wgan-gptarget_libri/model_other",

    )

    args = parser.parse_args()

    while len(sys.argv) > 1:
        sys.argv.pop()

    # Mono-audio only，仅支持单声道
    assert args.data_num_channels == 1

    # Save args，将上面的参数保存在本地txt文件中。 vars(obj)表示返回对象的 属性-属性值 组成dict; 下面语句中的sorted项是排序，不过key指定了按照每组值的第一项排序
    with open(os.path.join('./result', 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Make model kwarg dicts。 setattr(obj, "attr",
    # v)指为（类）对象中的一属性attr赋一个属性值v，具体参考：https://www.runoob.com/python/python-func-setattr.html
    # 在这里，表示给args对象添加属性'wavegan_g_kwargs'(与前面通过add_argument添加参数的方式类似)，然后为该属性赋值(即后面的词典)
    setattr(args, 'wavegan_g_kwargs', {
        'slice_len': args.data_slice_len,
        'nch': args.data_num_channels,
        # 'kernel_len': 15,  # args.wavegan_kernel_len
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        # 'use_norm': args.wavegan_batchnorm,
        'use_batchnorm': args.wavegan_batchnorm,
        'upsample': args.wavegan_genr_upsample
    })
    setattr(args, 'wavegan_d_kwargs', {
        # 'kernel_len': 25,  # args.wavegan_kernel_len
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        # 'use_norm': args.wavegan_batchnorm,
        'use_batchnorm': args.wavegan_batchnorm,
        'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
    })

    cur_time = time.strftime("%m-%d-%H-%M", time.localtime()) + "_" + args.wavegan_loss
    result_dir = os.path.join("./result/newsnr", cur_time)
    train_dir = os.path.join(result_dir, 'train')
    # validation_dir = os.path.join(result_dir, 'validation')
    summary_dir = os.path.join(result_dir, 'summary')
    model_dir = os.path.join(result_dir, 'model')
    train_tran_txt = os.path.join(result_dir, "train_tran.txt")
    # validation_tran_txt = os.path.join(result_dir, "validation_tran.txt")
    # validation_csv_dir = os.path.join(result_dir, 'validation_csv')
    # config_file = "/media/ps/data/gxy/Adv_audio/jasper_test.py"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    # if not os.path.exists(validation_dir):
    #     os.mkdir(validation_dir)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # if not os.path.exists(validation_csv_dir):
    #     os.mkdir(validation_csv_dir)

    # train(args, train_dir, validation_dir, summary_dir, model_dir, config_file, train_tran_txt, validation_tran_txt,
    #       result_dir)
    # train(args, train_dir, summary_dir, model_dir, config_file, train_tran_txt, result_dir)
    train(args, train_dir, summary_dir, model_dir, train_tran_txt, result_dir)


if __name__ == '__main__':
    mains()