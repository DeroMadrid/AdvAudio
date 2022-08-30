# coding=utf-8
from functools import reduce
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 2080ti有两个GPU，这里指定0，指仅使用0号gpu，写1指1号GPU，不写或者“0,1”都写指两个GPU都使用
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

import loader0_new3 as loader
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

# jasper and wav2letter+
from tf_jasper import compute_logfbank, idx2char, char2idx
from OpenSeq2Seq.open_seq2seq.utils.utils import get_base_config, create_model, \
    create_logdir, check_logdir, \
    check_base_model_logdir

sys.path.append("DeepSpeech")

def js_preds(decode) -> List[int]:
    """
    将decode这个稀疏张量转化为音频的转录文本
    :param decode: 是jasper的logits输出，解码后的稀疏张量[0]，是一个长度为1的list，decode.indices、dense_shape、values
    :return: 一个batch中音频文件的转录文本组成的list
    """
    res = np.zeros(decode.dense_shape) + len(idx2char) - 1
    for i in range(len(decode.values)):
        a, b = decode.indices[i]
        res[a, b] = decode.values[i]
    tran = ["".join(idx2char[int(x)] for x in y).replace("'", "") for y in res]
    return tran


def js_var(global_variables):
    return [x for x in global_variables if (x.name.startswith('F'))]


def train(args, train_dir_, validation_dir_, summary_dir_, model_dir_, config_file, train_tran_txt_, validation_tran_txt_,
          result_dir_):  # args是参数集

    """
    尽可能在跑代码之前，先将所有音频文件统一长度，所有转录也统一长度。
    """
    handle = tf.placeholder(tf.string, shape=[])  # 用于控制验证集和训练集的切换
    with tf.name_scope("loader"):
        # 这里需要注意loader0_new.py需要进行更改 只保留test的handle
        iterator, training_iterator, test_iterator = loader.get_iterator(
            handle, args.train_batch_size, args.validation_batch_size, args.training_wav_dir,
            args.training_tran_txt, args.validation_wav_dir, args.validation_tran_txt)

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

    ax = tf_frame_to_seq(y_frames=ax_f, org_len=x.shape[1])
    disturb = ax - x  # 扰动

    adv_input = ax[:, :, 0]  # shape(bs, 音频长度)
    fs = args.data_sample_rate  # 采样率
    random_noise = tf.random_normal(tf.shape(ax[:, :, 0]), stddev=2)
    batch_size = args.train_batch_size

    # ~~~~~~~~~~~~~~~~~~~~~~~deepspeech~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 将adv_input输入deepspeech-0.4模型，输出其logits，作为计算ctc-loss的inputs
    adv_input_int16 = adv_input * 32768 + random_noise
    deepspeech_pass_in = tf.clip_by_value(t=adv_input_int16, clip_value_min=-2 ** 15, clip_value_max=2 ** 15 - 1)
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        deepspeech_logits, seq_len = get_logits(new_input=deepspeech_pass_in)  # 调用deepspeech，返回logits值，用于计算ctc-loss
    ds_decodes, _ = tf.nn.ctc_beam_search_decoder(deepspeech_logits, seq_len, merge_repeated=False, beam_width=100)
    deepspeech_logits_lm = tf.transpose(tf.nn.softmax(deepspeech_logits), [1, 0, 2])  # 带语言模型解码时的输入
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

        # prediction
        lingvo_decoded = task.Decode(inputs)
    # ………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~jasper_loss~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    input_tf = tf.clip_by_value(ax[:, :, 0] * 32768 + random_noise, -2 ** 15, 2 ** 15 - 1)
    # 不需要进行归一化
    # gain = 1.0 / (np.max(np.abs(input_tf)) + 1e-5)
    # input_tf = input_tf * gain
    jasper_duration, jasper_features = compute_logfbank(input_tf)
    jasper_len = tf.constant(value=jasper_features.get_shape().as_list()[1], shape=(batch_size,), dtype=tf.int32)

    input_tensors = {}
    input_tensors["source_tensors"] = [jasper_features, jasper_len]

    arg = ['--config_file=' + config_file, '--mode=infer']
    args_js, base_config, base_model, config_module = get_base_config(arg)

    jasper_model = create_model(args_js, base_config, config_module, base_model, hvd=None, checkpoint=None)

    _outputs = [None] * jasper_model.num_gpus
    for gpu_cnt, gpu_id in enumerate(jasper_model._gpu_ids):
        with tf.device("/gpu:{}".format(gpu_id)), tf.variable_scope(
                # "",
                "ForwardPass",
                reuse=tf.AUTO_REUSE,
        ):
            # with tf.variable_scope("ForwardPass"):
            encoder_input = {"source_tensors": input_tensors["source_tensors"]}
            encoder_output = jasper_model.encoder.encode(input_dict=encoder_input)

            decoder_input = {"encoder_output": encoder_output}
            decoder_output = jasper_model.decoder.decode(input_dict=decoder_input)
            model_outputs = decoder_output.get("outputs", None)
            model_logits = decoder_output.get("logits", None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    generator_var = [x for x in tf.global_variables() if x.name.startswith('G')]

    # 若是保存ckpt的话，var_list对应的是仅你想保存的变量组成的list，不想保存的变量不用加入该list
    g_saver = tf.train.Saver(var_list=generator_var)

    # Run training
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)  # 用于创建会话时的参数配置
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 设置占用GPU80%的显存
    config.gpu_options.allow_growth = True  # 开始分配少量显存，然后按需增加

    with tf.Session(config=config) as sess:

        # 若要加载外部预训练模型（如，deepspeech），需要在全局变量初始化之后再加载（restore）；或者使用局部变量初始化方式。
        # 因为外部预训练模型的参数值已有，不需要初始化。
        sess.run(tf.global_variables_initializer())
        sess.run(test_iterator.initializer)

        training_handle = sess.run(training_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        # writer = tf.summary.FileWriter(logdir=summary_dir_, graph=sess.graph)  # 指定文件夹保存summary信息

        g_saver.restore(sess, save_path=tf.train.latest_checkpoint(args.model_path))
        # tensorboard --logdir=summury所在文件夹；  http://localhost:6006

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        snrs = 0
        step = 0
        val_loop = 0
        val_time = 0
        val_num = 200
        sum_snr = 0
        avg_snr = 0

        lv_same_all = 0
        lv_diff_all = 0
        ds_same_all = 0
        ds_diff_all = 0
        js_same_all = 0
        js_diff_all = 0
        val_start = time.time()

        ds_lm = True  # 若为True，解码deepspeech转录时带语言模型矫正

        while True:
            try:
                fetches = [x, ax, org_trans, ds_org_trans, deepspeech_logits_lm, ds_decodes[0], seq_len,
                           lv_org_trans, lingvo_decoded, js_org_trans, model_outputs[0]]
                val_r = sess.run(fetches, feed_dict={handle: test_handle})

                oa = val_r[0][:, :, 0] * 32768  # 原音频
                aa = val_r[1][:, :, 0] * 32768  # 对抗音频

                # 每**次迭代，查看转录结果
                dot = [str(i, encoding='utf-8').lower() for i in val_r[3]]  # deepspeech原音频转录
                lot = [str(i, encoding='utf-8').lower() for i in val_r[7]]  # lingvo原音频转录
                lat = [str(i, encoding='utf-8').lower() for i in val_r[8]['topk_decoded'][:, 0]]  # lingvo对抗转录

                if ds_lm:
                    dat = ds_preds_lm(val_r[4], val_r[6])  # deepspeech对抗音频转录
                else:
                    dat = ds_preds(val_r[5])

                jot = [str(i, encoding='utf-8').lower() for i in val_r[9]]  # jasper原始音频转录
                jat = js_preds(val_r[10])

                org = [str(i, encoding='utf-8').lower() for i in val_r[2]]
                # lv_same, lv_diff = str_list_cmp(lot, lat)  # 分别是存在相同单词的转录数，存在不同单词的数
                # ds_same, ds_diff = str_list_cmp(dot, dat)
                # js_same, js_diff = str_list_cmp(jot, jat)
                lv_same, lv_diff = str_list_cmp(org, lat)  # 分别是存在相同单词的转录数，存在不同单词的数
                ds_same, ds_diff = str_list_cmp(org, dat)
                js_same, js_diff = str_list_cmp(org, jat)

                lv_same_all += lv_same
                lv_diff_all += lv_diff
                ds_same_all += ds_same
                ds_diff_all += ds_diff
                js_same_all += js_same
                js_diff_all += js_diff

                fp = open(validation_tran_txt_, mode="a")
                # if val_loop == 0:
                    # txt_time = time.time() - start - val_time
                    # fp.write("step-{0}-{1:.5f}-{2}-{3}\n".format(step, txt_time, cur_same, cur_diff))
                for bs in range(args.validation_batch_size):
                    fp.write("step-{0}-{1}-{2}-org_ds: {3}\n".format(step, val_loop, bs, dot[bs]))
                    fp.write("step-{0}-{1}-{2}-adv_ds: {3}\n".format(step, val_loop, bs, dat[bs]))
                    fp.write("step-{0}-{1}-{2}-org_lv: {3}\n".format(step, val_loop, bs, lot[bs]))
                    fp.write("step-{0}-{1}-{2}-adv_lv: {3}\n".format(step, val_loop, bs, lat[bs]))
                    fp.write("step-{0}-{1}-{2}-org_js: {3}\n".format(step, val_loop, bs, jot[bs]))
                    fp.write("step-{0}-{1}-{2}-adv_js: {3}\n\n".format(step, val_loop, bs, jat[bs]))
                fp.close()

                for n in range(args.validation_batch_size):
                    valid_len = np.where(abs(oa[n, :]) > 0)[0][-1] + 1  # 返回该音频数组倒数第一个非0数值的索引+1
                    f_name_l = ['org_audio', 'adv_audio']

                    org_wav = np.array(np.clip(np.round(oa[n, :valid_len]), -2 ** 15, 2 ** 15 - 1),
                                   dtype=np.int16)
                    adv_wav = np.array(np.clip(np.round(aa[n, :valid_len]), -2 ** 15, 2 ** 15 - 1),
                                   dtype=np.int16)
                    org_wav = np.pad(org_wav, (10, 100), 'constant')  # 给其前后各填充10、100个0
                    adv_wav = np.pad(adv_wav, (10, 100), 'constant')

                    wav_snr = snr(org_wav, adv_wav)  # 信噪比
                    sum_snr += wav_snr

                    for fn in f_name_l:
                        audio_name = str(step) + "_" + str(val_loop) + '_' + str(n) + '_' + ''.join(org[n]) + ".wav"
                        f_name = os.path.join(validation_dir_, fn)
                        if not os.path.exists(f_name):
                            os.mkdir(f_name)
                        audio_data = adv_wav if 'adv' in fn else org_wav
                        write(filename=os.path.join(f_name, audio_name), rate=16000, data=audio_data)

                val_loop += 1

            except tf.errors.OutOfRangeError:
                break

        val_end = time.time()
        val_time += (val_end - val_start)
        val_num = val_loop * args.validation_batch_size
        avg_snr = sum_snr / val_num
        print("****** 训练集当前批次平均信噪比：", avg_snr)
        print("****** deepspeech失败：", ds_same_all)
        print("****** lingvo失败：", lv_same_all)
        print("****** jasper失败：", js_same_all)


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
        # training_wav_dir="/media/ps/data/gxy/Adv_audio/data/digit/uniform",
        # validation_wav_dir="/media/ps/data/gxy/Adv_audio/data/digit/uniform",  # 验证集音频文件所在目录
        training_wav_dir="/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/new_crop",
        validation_wav_dir="/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/new_crop",  # 验证集音频文件所在目录
        # training_tran_txt='./data/recaptchaV2/recaptcha5k/trantrain_d_l_j_w_p.txt',  # 训练集转录文件（测试时，只需将验证的路径改为测试集的）
        # training_tran_txt='/media/ps/data/gxy/Adv_audio/data/recaptchaV2/val/tran_d_l_j_p.txt',
        # training_tran_txt='/media/ps/data/gxy/Adv_audio/data/digit/tran_d_l_j_p.txt',
        # 训练集转录文件（测试时，只需将验证的路径改为测试集的）
        # validation_tran_txt='/media/ps/data/gxy/Adv_audio/data/digit/tran_d_l_j_p.txt',  # 验证集转录文件
        training_tran_txt='/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/tran_d_l_j_p.txt',
        validation_tran_txt='/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/tran_d_l_j_p.txt',  # 验证集转录文件

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

        validation_batch_size=8,
        # model_path="./result/recaptcha/11-09-09-14_wgan-gp/model",
        # model_path="./result/recaptcha/11-15-14-36_wgan-gp/model",
        # model_path="./result/recaptcha/11-15-14-36_wgan-gp/model",
        # model_path="./result/recaptcha/11-25-10-01_wgan-gp/model",
        model_path="./result/recaptcha/11-25-10-04_wgan-gp/model",
        # other_model_path="./result/recaptcha/03-24-21-56_wgan-gptarget_libri/model_other",

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

    # cur_time = time.strftime("%m-%d-%H-%M", time.localtime()) + "_" + args.wavegan_loss
    # result_dir = os.path.join("./result/recaptcha/11-11-16-02_wgan-gp")
    # result_dir = os.path.join("./result/recaptcha/11-15-10-58_wgan-gp")
    # result_dir = os.path.join("./result/recaptcha/11-15-14-36_wgan-gp")
    # result_dir = os.path.join("./result/recaptcha/11-25-10-01_wgan-gp")
    result_dir = os.path.join("./result/recaptcha/11-25-10-04_wgan-gp")
    validation_dir = os.path.join(result_dir, 'test')
    train_dir = os.path.join(result_dir, 'train')
    # validation_dir = os.path.join(result_dir, 'validation')
    summary_dir = os.path.join(result_dir, 'summary')
    model_dir = os.path.join(result_dir, 'model')
    train_tran_txt = os.path.join(result_dir, "train_tran.txt")
    validation_tran_txt = os.path.join(result_dir, "validation_tran.txt")
    # validation_csv_dir = os.path.join(result_dir, 'validation_csv')
    config_file = "/media/ps/data/gxy/Adv_audio/jasper_test.py"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # if not os.path.exists(validation_csv_dir):
    #     os.mkdir(validation_csv_dir)

    train(args, train_dir, validation_dir, summary_dir, model_dir, config_file, train_tran_txt, validation_tran_txt,
           result_dir)
    # train(args, train_dir, summary_dir, model_dir, config_file, train_tran_txt, result_dir)


if __name__ == '__main__':
    mains()