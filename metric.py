from scipy.io.wavfile import read, write
from pypesq import pesq
from glob import glob

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from jiwer import wer
import Levenshtein  # python自带的计算编辑距离的方法，用于与自己所写作比较， pip install python-Levenshtein
import librosa
from pystoi import stoi
import soundfile as sf


class Error(Exception):
    pass


def com_pesq(org_dir, adv_dir):
    """
    计算原音频和对抗音频之间的pesq分（用于评价语音质量感知）。-0.5到4.5分，4.5分最好，其中3.8被认为是传统电话网络中可以接收的语音
    质量，属于客观评价指标，又称为“话音清晰度指标”
    :param org_dir: 原音频文件所在的目录
    :param adv_dir: 对抗音频文件所在的目录
    :return: pesq分list
    """
    org_fps = glob(os.path.join(org_dir, "*"))
    adv_fps = glob(os.path.join(adv_dir, "*"))

    scores = []
    for i in range(len(org_fps)):
        sr, org_audio = read(org_fps[i])
        sr, adv_audio = read(adv_fps[i])

        # pesq(fs, ref, def, mode)，fs必须是8000或16000Hz，fs为8000时，只能用‘nb’；ref参考音频信号，def退化音频信号
        score = pesq(fs, org_audio, adv_audio, 'nb')  # 'wb'和'nb'分别指wide band 和 narrow band
        scores.append(score)
    return scores


# org_dir = "/home/abc/yzn/Audio_generate/adv_audio/result/train-1-1-1-1/lv_org_audio"
# adv_dir = "/home/abc/yzn/Audio_generate/adv_audio/result/train-1-1-1-1/lv_adv_audio"


def cmp_ser(tran_txt):
    """
    计算对抗转录与原转录之间是否存在不同的单词，若是存在，表示通过，否则不通过；相当于计算句错误率
    :param tran_txt: 保存有原转录和对抗转录的txt文件，格式：序号----:原转录:对抗转录
    :return: 总的通过率，通过的数量 / 总的数量
    """
    f = open(tran_txt, mode='r')
    lines = f.readlines()
    f.close()

    p = []
    for n in lines:
        n = n[:-1]  # 去掉最后的一个换行符
        org_tran, adv_tran = n.split(":")[1:]
        org_tran_list = org_tran.split()
        adv_tran_list = adv_tran.split()
        intersect = set(org_tran_list).intersection(set(adv_tran_list))  # 计算两个集合的交集
        # if len(intersect) == min(len(org_tran_list), len(adv_tran_list)):
        if org_tran_list == adv_tran_list:
            p.append(0)  # 不通过
        else:
            p.append(1)  # 通过
    r = sum(p) / len(p)
    return r


def cmp_wer(tran_txt):
    """
    计算原转录和对抗转录之间的词错误率WER。计算词错误率时，先把两个字符串中相同的位置对应上，然后再找之间的替换、删除、添加的词数。
    :param tran_txt: 保存有原转录和对抗转录的txt文件，格式：序号----:原转录:对抗转录
    :return: 所有转录的词错误率list
    """
    f = open(tran_txt, mode='r')
    lines = f.readlines()
    f.close()

    wers = []
    for n in lines:
        n = n[:-1]  # 去掉最后的一个换行符
        org_tran, adv_tran = n.split(":")[1:]
        org_tran_list = org_tran.split()
        adv_tran_list = adv_tran.split()
        a_l = []
        o_l = []
        q = 0
        for i in range(len(org_tran_list)):
            for j in range(q, len(adv_tran_list)):
                if adv_tran_list[j] == org_tran_list[i]:
                    a_l.append(j)
                    o_l.append(i)
                    q = j + 1
                    break
        a_l = [a_l[i] - a_l[i - 1] - 1 for i in range(1, len(a_l))]
        o_l = [o_l[i] - o_l[i - 1] - 1 for i in range(1, len(o_l))]
        sid_num = sum(map(lambda x: max(x[0], x[1]), zip(a_l, o_l)))  # 替换插入删除单词总数
        wer = sid_num / len(org_tran_list)
        wers.append(wer)
    return wers


def cmp_snr(org_dir, adv_dir):
    """
    计算原音频和对抗音频之间的信噪比。
    有用信号功率与噪声功率的比（此处功率为平均功率），也等于幅度比的平方
    :param org_dir: 原音频文件所在的目录
    :param adv_dir: 对抗音频文件所在的目录
    :return: 信噪比list
    """
    org_fps = glob(os.path.join(org_dir, "*"))
    adv_fps = glob(os.path.join(adv_dir, "*"))

    snrs = []
    for i in range(len(org_fps)):
        sr, org_audio = read(org_fps[i])
        sr, adv_audio = read(adv_fps[i])
        if len(org_audio) != len(adv_audio):
            raise Error("原音频和对抗音频的数据长度不一致！")
        sig_p = np.sum(np.square(org_audio))
        noi_p = np.sum(np.square(np.array(org_audio) - np.array(adv_audio)))
        snr = 10 * np.log10(sig_p / noi_p)


def min_dis(src, tgt):
    """
    精简版的最小编辑距离计算
    """
    src = " ".join(src.split())
    tgt = " ".join(tgt.split())
    # 若俩字符串均为空，返回0，若其中一个为空，返回另一个的长度
    if not (src and tgt):
        return len(src or tgt or "")

    n, m = len(src), len(tgt)

    # 初始化编辑距离矩阵，shape(n+1, m+1)，并且初始化第一行第一列的值，dis[0,0]=0， dis[i,0]=i， dis[0,j]=j
    dis = [[i + j for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if src[i - 1] == tgt[j - 1]:
                cost = 0
            else:
                cost = 1  # 这里将替换字符的代价设为1（有的代码中是2），插入和删除默认是1
            dis[i][j] = min(dis[i - 1][j] + 1, dis[i][j - 1] + 1, dis[i - 1][j - 1] + cost)
    return dis[-1][-1]


def min_dis_list(src_l, tgt_l):
    """
    计算两个元素数相等的list中的字符串的总的最小编辑距离
    """
    dis_sum = 0
    for i in range(len(src_l)):
        dis_sum += min_dis(src_l[i], tgt_l[i])
    return dis_sum


def str_cmp(str1, str2):
    """
    比较两个字符串，若是完全不相等，没有交集，返回1；若是完全相等，返回-1；否则返回0
    """
    str1 = str1.split()  # 会移除掉所有多出来的空格
    str2 = str2.split()
    intersect = set(str1).intersection(set(str2))  # 计算两个集合的交集
    if not intersect:
        return 1
    elif len(str1) == len(str2) and len(intersect) == len(str1):
        return -1
    else:
        return 0


def str_list_cmp(a, b):
    """
    #比较两个由字符串组成的list，返回“存在相同的个数”和“存在不相同的个数”
    """
    not_equal_sums = 0
    equal_sums = 0
    for i in range(len(a)):
        if str_cmp(a[i], b[i]) == 1:
            not_equal_sums += 1  # 完全不相同
        elif str_cmp(a[i], b[i]) == -1:
            equal_sums += 1  # 完全相同
    return len(a) - not_equal_sums, len(a) - equal_sums


def SegSNR(ref_wav, in_wav, windowsize, shift):
    """
    计算单个音频文件的分段信噪比。由于语音信号是一种缓慢变化的短时平稳信号，因而在不同时间段上的信噪比也应不一样。为了改善上面的问题，可以采用分段信噪比。
    分段信噪比即是先对语音进行分帧，然后对每一帧语音求信噪比，最好求均值。
    :param ref_wav: 噪声音频数据
    :param in_wav: 原音频数据
    :param windowsize: 分帧后的每一帧长度
    :param shift: 帧与帧之间的偏移长度
    :return: 平均分段信噪比
    """
    if len(ref_wav) == len(in_wav):
        pass
    else:
        print('音频的长度不相等!')
        minlenth = min(len(ref_wav), len(in_wav))
        ref_wav = ref_wav[: minlenth]
        in_wav = in_wav[: minlenth]
    # 每帧语音中有重叠部分，除了重叠部分都是帧移，overlap=windowsize-shift
    # num_frame = (len(ref_wav)-overlap) // shift
    #           = (len(ref_wav)-windowsize+shift) // shift
    num_frame = (len(ref_wav) - windowsize + shift) // shift  # 计算帧的数量
    print("print~ ", num_frame)

    SegSNR = np.zeros(num_frame)
    # 计算每一帧的信噪比
    for i in range(num_frame):
        noise_frame_energy = np.sum(ref_wav[i * shift: i * shift + windowsize] ** 2)  # 每一帧噪声的功率
        speech_frame_energy = np.sum(in_wav[i * shift: i * shift + windowsize] ** 2)  # 每一帧信号的功率
        # 存在问题：上述两值可能为0，导致不能相除
        if noise_frame_energy == 0:
            continue
        SegSNR[i] = np.log10(speech_frame_energy / noise_frame_energy)

    return 10 * np.mean(SegSNR)


def snr(org_audio, adv_audio):
    """
    计算（单个）原音频和对抗音频数据之间的信噪比。信噪比越大越好（说明噪声小）
    有用信号功率与噪声功率的比（此处功率为平均功率），也等于幅度比的平方
    :param org_dir: 原音频数据
    :param adv_dir: 对抗音频数据
    :return: 信噪比
    """
    org_audio = org_audio.astype(float)  # 若音频数据是int16类型，需要将其类型往上调，以便于计算它们的信噪比
    adv_audio = adv_audio.astype(float)
    if len(org_audio) != len(adv_audio):
        raise Error("原音频和对抗音频的数据长度不一致！")

    # 信噪比计算方法1：
    noise_norm = np.square(np.linalg.norm(adv_audio - org_audio))  # 噪声的2范数，然后平方
    if noise_norm == 0:
        return -1  # 两个音频相同
    org_norm = np.square(np.linalg.norm(org_audio))
    snr_ = 10 * np.log10(org_norm / noise_norm)

    # 信噪比计算方法2：
    # sig_p = np.sum(np.square(org_audio))
    # noi_p = np.sum(np.square(adv_audio - org_audio))
    # snr_1 = 10 * np.log10(sig_p / noi_p)
    return round(snr_, 2)


def psnr(org_audio, adv_audio):
    """峰值信噪比。表示信号的最大瞬时功率和噪声功率的比值，最大瞬时功率是语音数据中最大值的平方"""
    org_audio = org_audio.astype(float)  # 若音频数据是int16类型，需要将其类型往上调，以便于计算它们的信噪比
    adv_audio = adv_audio.astype(float)

    MSE = np.mean(np.square(adv_audio - org_audio))
    MAX = np.max(org_audio)
    psnr_ = 20 * np.log10(MAX / np.sqrt(MSE))
    return round(psnr_, 2)


def segsnr(org_audio, adv_audio, frame_len=512, shift=320):
    """分段信噪比。因为语音是短时平稳信号，所以先对语音分帧，然后计算每段的信噪比，最后求均值"""
    org_audio = org_audio.astype(float)  # 若音频数据是int16类型，需要将其类型往上调，以便于计算它们的信噪比
    adv_audio = adv_audio.astype(float)
    if len(org_audio) != len(adv_audio):
        raise Error("原音频和对抗音频的数据长度不一致！")

    segsnr_l = []
    noise_audio = adv_audio - org_audio
    size = len(org_audio)

    org_audio = np.pad(org_audio, (0, size + frame_len), mode='constant')
    noise_audio = np.pad(noise_audio, (0, size + frame_len), mode='constant')

    org_frame = np.stack([org_audio[j: j + frame_len] for j in range(0, size - shift, shift)], axis=-1)
    noise_frame = np.stack([noise_audio[j: j + frame_len] for j in range(0, size - shift, shift)], axis=-1)
    for i in range(0, len(org_frame)):
        org_f = np.sum(np.square(org_frame[i]))
        noi_f = np.sum(np.square(noise_frame[i]))
        snr_f = np.log10(org_f / noi_f)
        segsnr_l.append(snr_f)
    segsnr_ = 10 * np.mean(segsnr_l)
    return round(segsnr_, 2)


def _get_operation_counts(source_string, destination_string):
    """
    计算匹配的(一样的字符数)、替换的、删除的、插入的字符数
    """

    editops = Levenshtein.editops(source_string, destination_string)

    substitutions = sum(1 if op[0] == "replace" else 0 for op in editops)
    deletions = sum(1 if op[0] == "delete" else 0 for op in editops)
    insertions = sum(1 if op[0] == "insert" else 0 for op in editops)
    hits = len(source_string) - (substitutions + deletions)

    return hits, substitutions, deletions, insertions


def tran_wer(org, tgt):
    """
    计算原转录和对抗转录之间的词错误率。WER=（S+D+I）/(H+S+D)
    """
    org = " ".join(org.split())
    tgt = " ".join(tgt.split())
    error = wer(org, tgt)
    # or
    # h, s, d, i = _get_operation_counts(org, tgt)
    # error = float(s + d + i) / float(h + s + d)
    return error


def tran_wer_list(org_list, tgt_list):
    """
    计算两个list中的（字符串）原转录和对抗转录之间的词错误率。WER=（S+D+I）/(H+S+D)
    """
    for i in range(len(org_list)):
        org_list[i] = " ".join(org_list[i].split())
        tgt_list[i] = " ".join(tgt_list[i].split())
    error = wer(org_list, tgt_list)
    return error


def stoi_audio(org_audio, adv_audio, sr=16000):
    """
    STOI 反映人类的听觉感知系统对语音可懂度的客观评价，STOI 值介于0~1 之间，值越大代表语音可懂度越高，越清晰。
    :param org_audio: 数组，干净音频数组
    :param adv_audio: 数组， 对应的有噪声的音频数组，与干净音频的长度要一致，须得是1D
    :param sr: 两个音频的采样率
    :return:
    """
    stoi_score = stoi(x=org_audio, y=adv_audio, fs_sig=sr, extended=False)
    return stoi_score


def get_pesq(clean_wav, denoised_wav):
    """
    计算两个音频的pesq（语音质量感知评估），要求采样率为16000或8000，且采样率为8000时才能是窄带('nb')。
    mode参数可取值'wb'和'nb'。
    PESQ就是用经过处理后的语音文件（语音压缩、重构等）与原始语音进行比较。PESQ得分范围在-0.5--4.5之间。得分越高表示语音质量越好。
    git: https://github.com/vBaiCai/python-pesq
    :param clean_wav: 原始文件
    :param denoised_wav: 待评估文件
    :return: score
    """
    sr0, ref = read(clean_wav)
    sr1, deg = read(denoised_wav)

    # 检查两个音频文件长度，帧数相差不大于10
    if abs(len(ref) - len(deg)) > 10:
        raise Error("ref_wav/deg_wav两个音频长度不一致: %d/%d" % (len(ref), len(deg)))

    score = pesq(ref, deg, fs=16000, normalize=False)
    return score


def tran_acc(strs1, strs2):
    """
    计算两组字符串中相同的数量占总数的比例，即转录成功率。
    如果str1是一个字符串，表示有目标攻击中的目标字符串；否则为无目标攻击中原转录，数量与str2相同。
    """
    n = len(strs2)
    sums = 0
    if isinstance(strs1, str):
        for i in range(n):
            if strs2[i] == strs1:
                sums += 1
    else:
        for i in range(n):
            if strs1[i] == strs2[i]:
                sums += 1
    acc = sums / n
    return acc


def train_trend(txt):
    """
    根绝记录的数据结果绘制几种损失函数的变化情况以及信噪比变换情况
    :param txt: 存储记录结果的txt文件。
    """
    f = open(txt, mode='r')
    lines = f.readlines()
    f.close()

    G_loss, d_adv_loss, l_adv_loss, l2_loss, penalty_loss, noise, advsnr = [], [], [], [], [], [], []
    for line in lines:
        x.append(i)
        a0, a1, a2, a3, a4, a5, a6 = lines[i].strip().split("\t")
        G_loss.append(float(a0))
        d_adv_loss.append(float(a1))
        l_adv_loss.append(float(a2))
        l2_loss.append(float(a3))
        penalty_loss.append(float(a4))
        noise.append(float(a5))
        advsnr.append(float(a6))
    plt.plot(y=np.array(G_loss))


if __name__ == "__main__":
    # org_wav = "./temp/0.wav"
    # adv_wav = "./temp/1.wav"
    # sr, org_audio = read(org_wav)
    # sr, adv_audio = read(adv_wav)
    # org_audio = org_audio.astype(float)  # 若音频数据是int16类型，需要将其类型往上调，以便于计算它们的信噪比
    # adv_audio = adv_audio.astype(float)
    # r = get_pesq(org_wav, adv_wav)
    # print(r)
    # stoi_audio

    stios = 0
    s = 0
    prefix = "./result/recaptcha/test/03-20-19-41_wgan-gp/validation"
    org_path = os.path.join(prefix, "org_audio")
    adv_path = os.path.join(prefix, "adv_audio")
    org_wavs = glob(os.path.join(org_path, "*"))
    for i in range(len(org_wavs)):
        # sr, org_audio = read(org_wavs[i])
        # sr, adv_audio = read(os.path.join(prefix, "adv_audio", os.path.basename(org_wavs[i])))
        # org_audio = org_audio.astype(float)  # 若音频数据是int16类型，需要将其类型往上调，以便于计算它们的信噪比
        # adv_audio = adv_audio.astype(float)
        s += get_pesq(org_wavs[i], os.path.join(prefix, "adv_audio", os.path.basename(org_wavs[i])))

        # stios += stoi_audio(org_audio, adv_audio)
    stios = s / len(org_wavs)
    print(stios)