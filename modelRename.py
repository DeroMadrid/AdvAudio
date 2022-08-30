import tensorflow as tf

import argparse

import os

parser = argparse.ArgumentParser(description='')
parser.add_argument("--checkpoint_path", default='/media/ps/data/gxy/Adv_audio/w2lplus_model/model.ckpt-109800', help="restore ckpt") #原参数路径
parser.add_argument("--new_checkpoint_path", default='/media/ps/data/gxy/Adv_audio/w2lplus_model_new/', help="path_for_new ckpt") #新参数保存路径
parser.add_argument("--add_prefix", default='WaveLetter/', help="prefix for addition") #新参数名称中加入的前缀名
args = parser.parse_args()

def main():

    if not os.path.exists(args.new_checkpoint_path):
        os.makedirs(args.new_checkpoint_path)
    with tf.Session() as sess:
        new_var_list=[] #新建一个空列表存储更新后的Variable变量
        for var_name, _ in tf.contrib.framework.list_variables(args.checkpoint_path): #得到checkpoint文件中所有的参数（名字，形状）元组
            var = tf.contrib.framework.load_variable(args.checkpoint_path, var_name)
            new_name = var_name
            # new_name = var_name.split("/")[1]
            new_name = args.add_prefix + new_name #在这里加入了名称前缀，大家可以自由地作修改
            #除了修改参数名称，还可以修改参数值（var）
            print('Renaming %s to %s.' % (var_name, new_name))
            renamed_var = tf.Variable(var, name=new_name) #使用加入前缀的新名称重新构造了参数
            new_var_list.append(renamed_var) #把赋予新名称的参数加入空列表
        print('starting to write new checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list) #构造一个保存器
        sess.run(tf.global_variables_initializer()) #初始化一下参数（这一步必做）
        model_name = 'model.ckpt-109800' #构造一个保存的模型名称
        checkpoint_path = os.path.join(args.new_checkpoint_path, model_name) #构造一下保存路径
        saver.save(sess, checkpoint_path) #直接进行保存
        print("done !")

if __name__ == '__main__':
    main()
