"""
尝试重写一下OpenSeq2Seq特征提取的部分并进行调用测试
原版本：在numpy格式的音频数据下进行特征提取等input_tensor的制作
现版本：需要直接用tensor测试
参考deepspeech call部分的代码进行重写
"""
import numpy as np
import tensorflow as tf
from glob import glob
import scipy.io.wavfile as wav
from typing import List
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import sys
import io
from tf_jasper import compute_logfbank, idx2char
from OpenSeq2Seq.open_seq2seq.utils.utils import get_base_config, create_model,\
                                     create_logdir, check_logdir, \
                                     check_base_model_logdir

sys.path.append("OpenSeq2Seq")


def w2l_preds(decode) -> List[int]:
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

"""
这部分代码对于输入的音频进行特征提取 然后构建模型 并定义CTC损失节点
然后开一个session调用特征提取和模型 得到预测序列和loss
"""
def call_w2l(config_file, audios_, tgts_, audio_data_maxlen_, tran_save_txt_):
    batch_size = 1

    # 创建模型得到参数
    arg = ['--config_file='+config_file, '--mode=infer']
    args, base_config, base_model, config_module = get_base_config(arg)

    """
    # 加载模型
    load_model = base_config.get('load_model', None)
    restore_best_checkpoint = base_config.get('restore_best_checkpoint', False)
    base_ckpt_dir = check_base_model_logdir(load_model, args,
                                            restore_best_checkpoint)
    base_config['load_model'] = base_ckpt_dir
    # 不使用Horovod
    hvd = None
    """

    w2l_model = create_model(args, base_config, config_module, base_model, hvd=None, checkpoint=None)
    # if 'initializer' not in jasper_model.params:
    #     initializer = None
    # else:
    #     init_dict = jasper_model.params.get('initializer_params', {})
    #     initializer = jasper_model.params['initializer'](**init_dict)

    # 看送入模型的input_tensor需要哪些 并按照需求进行占位
    input_tf = tf.placeholder(dtype=tf.float32, shape=[batch_size, audio_data_maxlen_], name="audio_inputs")
    tgt_tf = tf.placeholder(dtype=tf.string, shape=[batch_size], name="orgTran_or_filename")

    # 此处得到features，参考deepspeech对tensor类型音频进行logfban的过程 *可以另开一个file写一个关于特征提取的函数
    w2l_duration, w2l_features = compute_logfbank(input_tf)
    # jasper_len = tf.int32(len(jasper_features))
    # jasper_features.set_shape([jasper_model._params['batch_size'], None,
    #                            jasper_model._params['num_audio_features']])
    # jasper_len = tf.reshape(jasper_len, [jasper_model._params['batch_size']])
    w2l_len = tf.constant(value=w2l_features.get_shape().as_list()[1], shape=(batch_size,), dtype=tf.int32)
    # jasper_len = tf.to_int32(jasper_features[:, 0, 0])

    input_tensors ={}
    input_tensors["source_tensors"] = [w2l_features, w2l_len]
    # 得到的input_tensors应该包含source_sequence (shape=[batch_size x sequence length x num_audio_features])
    # source_length (shape=[batch_size])
    # print("source_sequence", jasper_features.shape, "  src_length:", jasper_len.shape)

    # 产生logits和预测结果
    # 注意dump_outputs要修改为true
    _outputs = [None] * w2l_model.num_gpus
    # results_per_batch = []
    for gpu_cnt, gpu_id in enumerate(w2l_model._gpu_ids):
        with tf.device("/gpu:{}".format(gpu_id)), tf.variable_scope(
                "ForwardPass",
                # name_or_scope=tf.get_variable_scope(),
                # re-using variables across GPUs.
                reuse=tf.AUTO_REUSE,
                # initializer=initializer,
                # dtype=jasper_model.get_tf_dtype(),
        ):
            # 这里的loss是有问题的,loss=None
            # loss, _outputs[gpu_cnt], _logits = jasper_model._build_forward_pass_graph(
            #     input_tensors,
            #     gpu_id=gpu_cnt
            # )

            # _outputs[gpu_cnt] = tf.sparse_tensor_to_dense(_outputs[gpu_cnt])
            # print("***************_outputs[gpu_cnt][0]", _outputs[gpu_cnt][0])
            # print(idx2char)
            # print("_outputs[gpu_cnt]:", _outputs[gpu_cnt][0])
            # decoded_text = sparse_tensor_to_chars(_outputs[gpu_cnt][0], idx2char)
            # results_per_batch.append(jasper_model.infer(input_tensors, _outputs[gpu_cnt]))

            # with tf.variable_scope("ForwardPass"):
            encoder_input = {"source_tensors": input_tensors["source_tensors"]}
            encoder_output = w2l_model.encoder.encode(input_dict=encoder_input)

            decoder_input = {"encoder_output": encoder_output}
            decoder_output = w2l_model.decoder.decode(input_dict=decoder_input)
            model_outputs = decoder_output.get("outputs", None)
            model_logits = decoder_output.get("logits", None)
            # decoded_text = sparse_tensor_to_chars(model_outputs[0], idx2char)
            # 转换为文本这部分可以参考deepspeech的方法



    """
    js_total_loss = tf.nn.ctc_loss(
        labels=dense_to_sparse(tgt_sequence, tgt_length),
        inputs=_logits,
        sequence_length=jasper_len,
        ignore_longer_outputs_than_inputs=True,
    )
    """

    # 设置GPU、模型等相关参数
    w2l_saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)

    # 开启Session 运行各个tensorflow节点 得到结果并且保存
    with tf.Session(config=config) as sess:
        # latest_ckpt_js = tf.train.latest_checkpoint("/media/ps/data/gxy/Adv_audio/jasper_model/model.ckpt-438235")
        w2l_saver.restore(sess, save_path = "./w2lplus_model/model.ckpt-109800")

        loops = len(audios_) // batch_size

        for i in range(loops):
            p1, p2 = i*batch_size, (i+1)*batch_size
            audio_bs = np.array(audios_[p1: p2], dtype=float)
            tgt_bs = tgts_[p1: p2]
            # print("*****************audio",audio_bs,"*****************tgt", tgt_bs)
            # tgt_np, decode, logits = sess.run([tgt_tf, _outputs[gpu_cnt], _logits],
            #                                   feed_dict={input_tf: audio_bs, tgt_tf: tgt_bs})

            input, tgt_np, output, logits= sess.run([input_tensors, tgt_tf, model_outputs[0], model_logits],
                                              feed_dict={input_tf: audio_bs, tgt_tf: tgt_bs})
            # input, tgt_np, encode, decode = sess.run([input_tensors, tgt_tf, encoder_output, model_outputs],
            #                                  feed_dict={input_tf: audio_bs, tgt_tf: tgt_bs})
            # print("_outputs[gpu_cnt]:", _outputs[gpu_cnt])


            # decode = sparse_tensor_to_chars(_outputs[gpu_cnt][0], idx2char)
            # decode = _decode[0]

            decode = w2l_preds(output)

            fp = open(tran_save_txt_, mode='a')  # 将原始转录文本和deepspeech的转录文本保存到该txt文件中
            for r in range(len(decode)):
                if isinstance(tgt_np[r], bytes):
                    tgt_np[r] = str(tgt_np[r], encoding='utf-8')
                print("{0}:\t{1}".format(tgt_np[r], decode[r]))
                fp.write("{0}:\n{1}\n".format(tgt_np[r], decode[r]))
            fp.close()
            # print("logits:",logits)
    return

def w2l_main(config_file, audio_dir_, tran_txt_=None, tran_j_txt_=None):
    """
    :param audio_dir_: 需要转录的音频文件目录
    :param result_txt: 保存jasper的转录结果，格式：'音频文件名:jasper转录'
    """

    audios = []
    tgts = []

    audio_path_list = glob(os.path.join(audio_dir_, '*'))
    # print("******************audio_path_list", audio_path_list)
    for audio_path in audio_path_list:
        # fname = os.path.basename(audio_path).split(".")[0]
        fname = os.path.basename(audio_path)
        sr, audio = wav.read(audio_path)
        # 先需要对音频进行归一化 float32->[-1,1]
        # audio = audio.astype(np.float32)
        # gain = 1.0 / (np.max(np.abs(audio)) + 1e-5)
        # audio = audio * gain
        #----完成归一化——————————————————————————————
        # print("*******************************audio:", audio, "  fname:", fname)
        audios.append(audio)
        tgts.append(fname)

    # print("*************************************", audios, tgts)

    audio_data_maxlen = max(map(len, audios))
    audios = [np.pad(i, pad_width=(0, audio_data_maxlen - len(i)), mode="constant") for i in audios]

    call_w2l(config_file, audios, tgts, audio_data_maxlen, tran_save_txt_= result_txt)


if __name__ == "__main__":
    # audio_dir = "/media/ps/data/gxy/Adv_audio/result/recaptcha/11-29-10-48_wgan-gp/test/adv_audio"
    # result_txt = "/media/ps/data/gxy/Adv_audio/result/recaptcha/11-29-10-48_wgan-gp/test/adv_w2l.txt"
    # audio_dir = "/media/ps/data/gxy/Adv_audio/result/recaptcha/11-29-10-45_wgan-gp/test/adv_audio"
    # result_txt = "/media/ps/data/gxy/Adv_audio/result/recaptcha/11-29-10-45_wgan-gp/test/adv_w2l.txt"
    # audio_dir = "/media/ps/data/gxy/Adv_audio/result/recaptcha/11-25-10-01_wgan-gp/test/adv_audio"
    # result_txt = "/media/ps/data/gxy/Adv_audio/result/recaptcha/11-25-10-01_wgan-gp/test/adv_w2l.txt"
    audio_dir = "/media/ps/data/gxy/Adv_audio/result/recaptcha/11-25-10-04_wgan-gp/test/adv_audio"
    result_txt = "/media/ps/data/gxy/Adv_audio/result/recaptcha/11-25-10-04_wgan-gp/test/adv_w2l.txt"
    config_file = "/media/ps/data/gxy/Adv_audio/w2l_test.py"
    w2l_main(config_file, audio_dir, result_txt)
