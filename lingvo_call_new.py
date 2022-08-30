"""
lingvo自动语音识别系统的测试
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf
from lingvo import model_imports
from lingvo import model_registry
import numpy as np
import scipy.io.wavfile as wav
from tool import create_features, create_inputs
import time
from lingvo.core import cluster_factory
from glob import glob
from digit_rectify import digit_lm


def lingvoCall(audios_np_, tgt_np_,  sr_, masks_freq_, audio_data_maxlen_, freq_maxlen_,
               batch_size_, tran_save_txt_):

    tf.set_random_seed(1234)   # 使用它，可以使得所有随机函数跨session生成相同的随机数
    tfconf = tf.ConfigProto(allow_soft_placement=True)
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

        input_tf = tf.placeholder(dtype=tf.float32, shape=[batch_size_, audio_data_maxlen_], name='audios_input')
        tgt_tf = tf.placeholder(dtype=tf.string, shape=[batch_size_], name='org_tran')
        sample_rate_tf = tf.placeholder(dtype=tf.int32, shape=None, name='sr')
        mask_tf = tf.placeholder(dtype=tf.float32, shape=[batch_size_, freq_maxlen_, 80], name='mask_freq')

        # generate the features and inputs
        features = create_features(input_tf, sample_rate_tf, mask_tf)
        inputs = create_inputs(model, features, tgt_tf, batch_size_, mask_tf)

        # loss
        metrics = task.FPropDefaultTheta(inputs)
        loss = tf.get_collection("per_loss")[0]

        # prediction
        decoded_outputs = task.Decode(inputs)

        saver = tf.train.Saver()

        with tf.Session(config=tfconf) as sess:
            saver.restore(sess, save_path="./lingvo_model/ckpt-00908156")
            dec_metrics_dict = task.CreateDecoderMetrics()

            loops = len(audios_np_) // batch_size_
            for i in range(loops):
                print(i)
                p1, p2 = i*batch_size_, i*batch_size_+batch_size_
                audio_bs = audios_np_[p1: p2]
                tgt_bs = tgt_np_[p1: p2]
                mask_bs = masks_freq_[p1: p2]
                fd = {input_tf: audio_bs, tgt_tf: tgt_bs, mask_tf: mask_bs, sample_rate_tf: sr_}
                predictions, tgts = sess.run([decoded_outputs, tgt_tf], feed_dict=fd)
                task.PostProcessDecodeOut(predictions, dec_metrics_dict)

                fp = open(tran_save_txt_, mode='a')  # 用于保存原转录和lingvo转录的txt文件
                for k in range(batch_size_):
                    if isinstance(tgts[k], bytes):
                        tgts[k] = str(tgts, encoding='utf-8')
                    pred = predictions['topk_decoded'][k, 0]
                    if isinstance(pred, bytes):
                        pred = str(pred, encoding='utf-8').lower().strip()
                    pred = '' if '<' in pred else pred
                    # pred = digit_lm(pred)
                    # print("{0}:\t{1}:\t{2}:\t{3}".format(tgts[k], *tran_d_[tgts[k]], pred))  # 名字:名字：ds转录：lv转录
                    # fp.write("{0}:{1}:{2}:{3}\n".format(tgts[k], *tran_d_[tgts[k]], pred))
                    print("{0}:\t{1}".format(tgts[k], pred))  # 名字:名字：ds转录：lv转录
                    fp.write("{0}:\n{1}\n".format(tgts[k], pred))
                fp.close()


# def lv_main(tran_d_txt_, audio_dir_, tran_d_l_txt_):
def lv_main(audio_dir_, tran_l_txt_):
    audio_path_list = glob(os.path.join(audio_dir_, "*"))

    # f = open(tran_d_txt_, mode='r')
    # lines = f.readlines()
    # f.close()
    # tran_d = {}
    # for line in lines:
    #     name, org_tran, ds_tran = line.strip().split(':')
    #     tran_d[name] = (org_tran, ds_tran)

    audios, tgts = [], []

    for k in audio_path_list:
        fname = os.path.basename(k).split(".")[0]
        sr, audio = wav.read(k)
        if max(audio) < 1:
            audio *= 32768
        audios.append(audio)
        tgts.append(fname)

    lengths = list(map(len, audios))
    audio_len = len(audios)
    batch_size = 1

    # 如果出现这样的错误：ValueError: Dimensions must be equal, but are 297 and 294 for
    # 'mul_44' (op: 'Mul') with input shapes: [11,297,80], [11,294,80],
    # 就将max_length人为的调大一些，然后尝试，找到一个不出现该错误的值
    # audio_data_maxlen = max(lengths) + 100  #
    audio_data_maxlen = max(lengths) + 200 
    lengths_freq = (np.array(lengths) // 2 + 1) // 240 * 3
    freq_maxlen = (audio_data_maxlen // 2 + 1) // 240 * 3  # 该值相当于计算音频的mfcc的分帧时的帧数
    masks_freq = np.zeros([audio_len, freq_maxlen, 80], dtype=float)

    audios_np = np.zeros([audio_len, audio_data_maxlen], dtype=float)
    for i in range(audio_len):
        audios_np[i, :lengths[i]] = audios[i]
        masks_freq[i, :lengths_freq[i], :] = 1

    tgt_np = np.array(tgts)

    # lingvoCall(audios_np, tgt_np, tran_d, sr, masks_freq, audio_data_maxlen, freq_maxlen, batch_size, tran_d_l_txt_)
    lingvoCall(audios_np, tgt_np, sr, masks_freq, audio_data_maxlen, freq_maxlen, batch_size, tran_l_txt_)


if __name__ == "__main__":
    #tran_d_txt = './data/librspeech_other-4/tran_d.txt'    # 格式，音频文件名子:音频文件名字:deepspeech转录
    #audio_dir = "./data/librspeech_other-4/train_wav"   # 音频文件夹
    #tran_d_l_txt = "./data/librspeech_other-4/tran_d_l.txt"  # 得到的新的文件格式，音频文件名子：音频文件名字：ds转录:lv转录
    # tran_d_txt = '/media/ps/data/YZN/Adv_audio/data/recaptchaV2/recaptcha5k/tranval_d.txt'
    # audio_dir = "/media/ps/data/YZN/Adv_audio/data/recaptchaV2/recaptcha5k/val"
    # tran_d_l_txt = "/media/ps/data/YZN/Adv_audio/data/recaptchaV2/recaptcha5k/tranval_d_l.txt"

    # tran_d_txt = '/media/ps/data/gxy/Adv_audio/data/recaptchaV2/val/tran_d.txt'
    # audio_dir = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/val/wav"
    # tran_d_l_txt = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/val/tran_d_l.txt"
    # lv_main(tran_d_txt, audio_dir, tran_d_l_txt)

    # tran_d_txt = '/media/ps/data/gxy/Adv_audio/data/recaptchaV2/org200/tran_d.txt'
    # audio_dir = "/media/ps/data/gxy/Adv_audio/result/recaptcha/12-21-09-38_wgan-gp/test_old/adv_audio"
    # tran_l_txt = "/media/ps/data/gxy/Adv_audio/result/recaptcha/12-21-09-38_wgan-gp/test_old/advuntarget_lingvo.txt"
    audio_dir = "/media/ps/data/YZN/Adv_audio2/result/recaptcha/07-21_wgan-gptarget/07-23-15-08_wgan-gp/test_old/org_audio"
    tran_l_txt = "/media/ps/data/YZN/Adv_audio2/result/recaptcha/07-21_wgan-gptarget/07-23-15-08_wgan-gp/test_old/orguntarget_lingvo.txt"
    # lv_main(tran_d_txt, audio_dir, tran_d_l_txt)
    lv_main(audio_dir, tran_l_txt)


