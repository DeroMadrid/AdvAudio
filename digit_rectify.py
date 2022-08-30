def edit_distance(str1, str2):
    """
    计算两个字符串之间的编辑距离
    """
    n1, n2 = len(str1), len(str2)
    edit = [[i + j for j in range(n2 + 1)] for i in range(n1 + 1)]
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(edit[i-1][j]+1, edit[i][j-1]+1, edit[i-1][j-1]+d)
    return edit[n1][n2]


def digit_words():
    """纯数字单词list"""
    return ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero']


def closest_word(word, words=None):
    """计算无效单词与其他所有单词的编辑距离，返回编辑距离最小的有效单词用于替代该无效单词"""
    if word in words:
        return word
    min_dis = 5
    closest = None
    for i in words:
        dis = edit_distance(word, i)
        if min_dis > dis:
            closest = i
            min_dis = dis
    return closest


def digit_lm(tran_list):
    """对数字转录结果(list)进行遍历，将其中存在的无效单词转化为有效单词"""
    for t in range(len(tran_list)):
        word_list = tran_list[t].split()
        for i in range(len(word_list)):
            word_list[i] = closest_word(word_list[i], digit_words())
        tran_list[t] = ' '.join(word_list)
    return tran_list


if __name__ == "__main__":
    a = ["on two three", "nin eiht"]
    t = digit_lm(a)
    for i in t:
        print(i)

