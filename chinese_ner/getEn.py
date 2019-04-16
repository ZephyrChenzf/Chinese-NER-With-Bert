import numpy
import copy


def getEnt(test_str_list, result):
    '''

    :param test_str_list: 预测字符
    :param result: 预测标签
    :return:
    '''
    dic = {}
    words = []
    tags = []
    poss = []
    if True:
        temp_w = copy.copy(test_str_list)
        temp_t = result

        temp_w.append('end')
        temp_t.append('B-end')
        word = ['start']
        tag = 'start'
        pos = [0]
        for i in range(len(temp_w)):
            if (temp_t[i][0] == 'B' or temp_t[i][0] == 'S'):
                if len(word) != 0:
                    pos.append(pos[0] + len(word))
                    words.append(''.join(word))
                    poss.append(pos)
                    tags.append(tag)
                    word = []
                    pos = []
                    tag = temp_t[i][2:]
                    word.append(temp_w[i])
                    pos.append(i)
            elif temp_t[i][0] == 'I':
                word.append(temp_w[i])
            else:
                pass
        for i in range(len(words)):
            word = words[i]
            tag = tags[i]
            pos = poss[i]
            if tag not in dic:
                dic[tag] = []
                dic[tag].append((word, pos))  # word and pos
            else:
                dic[tag].append((word, pos))
    dic.pop('start')
    if len(dic):
        for key, value in dic.items():
            # if key == 'start':
            #     continue
            print(key + ':'+value.__str__(),end='\r')
        print('\n')
    else:
        print('nothing..')
    # dic.pop('start')
    # return dic

# if __name__=='__main__':
#     getEnt('./data/result_mytestdata')

