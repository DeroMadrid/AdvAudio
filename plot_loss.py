import numpy as np
import matplotlib.pyplot as plt
import os


def plot_loss(txt_path, k=10):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    iter = []   # 迭代次数
    ds_loss = []  # deepspeech对抗损失
    lv_loss = []  # lingvo对抗损失
    global_loss = []
    for i in range(len(lines)):
        line = lines[i].strip().split("\t")
        line = list(map(float, line))
        gl, ds_al, lv_al, l2l, pl, _, _ = line
        iter.append(i*100)
        ds_loss.append(ds_al)
        lv_loss.append(lv_al)
        global_loss.append(gl+5*ds_al+10*lv_al+3*l2l+pl)
    # plt.plot(iter, ds_loss, color="green", label='deepspeech_advloss')
    # plt.plot(iter, lv_loss, color="red", label='lingvo_advloss')
    # plt.plot(iter, global_loss, color="blue", label='global_loss')
    # plt.legend()
    # plt.show()

    # 为了便于比较画图，每项都只取其前k项
    return iter[:k], ds_loss[:k], lv_loss[:k], global_loss[:k]

# path = r"E:\实验室学习\本组工作\yzn\lsganT.txt"
# plot_loss(path)


if __name__ == "__main__":
    wgan_gp_path = r"E:\实验室学习\本组工作\yzn\wgan-gpT.txt"
    wgan_path = r"E:\实验室学习\本组工作\yzn\wganT.txt"
    lsgan_path = r"E:\实验室学习\本组工作\yzn\lsganT.txt"
    dcgan_path = r"E:\实验室学习\本组工作\yzn\dcganT.txt"
    wgan_gps = plot_loss(wgan_gp_path, 64)    # 总共4项，分别是迭代次数、lingvo对抗损失、deepspeech对抗损失、全局损失
    wgans = plot_loss(wgan_path, 64)
    lsgans = plot_loss(lsgan_path, 64)
    dcgans = plot_loss(dcgan_path, 64)
    # print(len(wgan_gps[0]))
    # print(len(wgans[0]))
    # print(len(lsgans[0]))
    # print(len(dcgans[0]))

    i = 3
    d = {1: "deepspeech_loss", 2: "lingvo_loss", 3: "global_loss"}
    plt.plot(wgan_gps[0], wgan_gps[i], color="green", label='wgan-gp')
    plt.plot(wgan_gps[0], wgans[i], color="red", label='wgan')
    plt.plot(wgan_gps[0], lsgans[i], color="blue", label='lsgan')
    plt.plot(wgan_gps[0], dcgans[i], color="skyblue", label='dcgan')
    plt.xlabel('iteration times')
    plt.ylabel(d[i])
    plt.legend()
    # plt.show()
    prefix = r"E:\实验室学习\本组工作\yzn"

    save_path = os.path.join(prefix, "{0}_target.png".format(d[i]))
    plt.savefig(save_path)






