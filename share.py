"""
该文件主要包含一些多个程序会调用到的函数
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from typing import List
from util.text import Alphabet
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer, ctc_beam_search_decoder_batch, ctc_beam_search_decoder


def toks():
    """词典"""
    return " abcdefghijklmnopqrstuvwxyz'-"


def ds_var(global_variables):
    return [x for x in global_variables if ((x.name.startswith('h') and (len(x.name) <= 4)) or
                                          (x.name.startswith('b') and (len(x.name) <= 4)) or
                                          x.name.startswith('lstm'))]


def ds_preds_lm(probs_seq, seq_lengths) -> List[int]:
    """
    输入是deepspeech的logits和seq_len，输出是其预测结果（使用语言模型）
    :param probs_seq: 三维数组，shape(bs, 帧长, 29)
    :param seq_lengths: shape(bs,)
    :return: list，deepspeech的预测结果
    """
    lm_alpha = 0.75
    lm_beta = 1.85
    lm_trie_path = "./deepSpeech_model/deepspeech-0.6.1-models/trie"
    lm_binary_path = "./deepSpeech_model/deepspeech-0.6.1-models/lm.binary"
    alphabet_config_path = "./deepSpeech_model/deepspeech-0.6.1-models/alphabet.txt"
    alphabet = Alphabet(os.path.abspath(alphabet_config_path))
    scorer = Scorer(lm_alpha, lm_beta, lm_binary_path, lm_trie_path, alphabet)
    ds_decoded = ctc_beam_search_decoder_batch(probs_seq=probs_seq, seq_lengths=seq_lengths, alphabet=alphabet,
                                               beam_size=500, num_processes=2, scorer=scorer,
                                               cutoff_prob=1.0,
                                               cutoff_top_n=40)
    pred = [ds_decoded[i][0][1].lower() for i in range(len(ds_decoded))]
    return pred


def ds_preds(decode) -> List[int]:
    """
    将decode这个稀疏张量转化为音频的转录文本
    :param decode: 是deepspeech的logits输出，解码后的稀疏张量[0]，是一个长度为1的list，decode.indices、dense_shape、values
    :return: 一个batch中音频文件的转录文本组成的list
    """
    res = np.zeros(decode.dense_shape) + len(toks()) - 1
    for i in range(len(decode.values)):
        a, b = decode.indices[i]
        res[a, b] = decode.values[i]
    tran = ["".join(toks()[int(x)] for x in y).replace("-", "") for y in res]
    return tran