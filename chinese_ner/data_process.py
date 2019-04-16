"""
@autor: chenzf
@file: data_process.py
@time: 2019/4/9 4:57 PM

"""
import sys,os
sys.path.append(os.path.abspath('..'))

from pytorch_pretrained_bert import BertTokenizer
def tokens2id(tokenizer,text):
    tokenized_text = tokenizer.tokenize(text)
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens

def get_data(dataPath,maxLen,output_path):
    all_data={}
    # train_data_path=dataPath+'train_data'
    # test_data_path=dataPath+'test_data'
    train_data_path=dataPath
    tokenizer=BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)
    tag2inx={'[PAD]':0}
    inx2tag={0:'[PAD]'}

    with open(train_data_path,'r') as data:
        train_x = []  # 总x训练集
        train_y = []  # 总y训练集
        train_mask=[]
        sen_x = []  # 每次存一句话的id组
        sen_y = []  # 每次存一句话的标签id组
        count=0
        for line in data:
            line = line.strip()
            if (line == "" or line == "\n" or line == "\r\n"):  # 一句话结束了
                # tokenized_text = tokenizer.tokenize(sen_x)
                tokenized_text=sen_x
                assert len(tokenized_text)==len(sen_y)
                # 开始对每句话进行裁剪，主要是最大长度的限制
                length=len(tokenized_text)
                while length>maxLen-2:
                    flag=False
                    for j in reversed(range(maxLen-2)):#反向访问，99、98、97...
                        if tokenized_text[j]=='，' or tokenized_text[j]=='、':
                            x_cut=tokenizer.convert_tokens_to_ids(['[CLS]']+ tokenized_text[:j+1] + ['[SEP]'])
                            true_len=len(x_cut)
                            x_cut += [0] * (maxLen - true_len)
                            y_cut=[tag2inx['O']]+sen_y[:j+1]+[tag2inx['O']]
                            y_cut+=[0] * (maxLen - true_len)
                            mask_cut = [1] * true_len + [0] * (maxLen - true_len)
                            assert len(x_cut)==len(y_cut)==len(mask_cut)
                            train_x.append(x_cut)
                            train_y.append(y_cut)
                            train_mask.append(mask_cut)
                            tokenized_text=tokenized_text[j+1:]
                            sen_y=sen_y[j+1:]
                            length = len(tokenized_text)
                            break
                        if j==0:
                            flag=True
                        if flag:
                            x_cut=tokenizer.convert_tokens_to_ids(['[CLS]']+ tokenized_text[:maxLen-2] + ['[SEP]'])
                            y_cut=[tag2inx['O']]+sen_y[:maxLen-2]+[tag2inx['O']]
                            mask_cut=[1]*len(x_cut)
                            assert len(x_cut) == len(y_cut) == len(mask_cut)
                            train_x.append(x_cut)
                            train_y.append(y_cut)
                            train_mask.append(mask_cut)
                    if flag:break

                if length <= maxLen-2:
                    x_cut=tokenizer.convert_tokens_to_ids(['[CLS]']+ tokenized_text + ['[SEP]'])
                    # 同时mask
                    true_len = len(x_cut)
                    x_cut+=[0]*(maxLen-true_len)
                    y_cut=[tag2inx['O']]+sen_y+[tag2inx['O']]
                    y_cut+=[0]*(maxLen-true_len)
                    mask_cut=[1]*true_len+[0]*(maxLen-true_len)
                    assert len(x_cut) == len(y_cut)== len(mask_cut)
                    train_x.append(x_cut)
                    train_y.append(y_cut)
                    train_mask.append(mask_cut)
                sen_x = []
                sen_y = []
                # if count==25:
                #     print('..')
                count+=1
                if count%1000==0:
                    print('{} sentences has been processed...'.format(count))
                continue
            line = line.split(' ')
            if (len(line) < 2):
                continue
            sen_x.append(line[0])
            if line[1] in tag2inx:  # 同理，注意不同标签对应的id与初始碰到的标签有关
                sen_y.append((tag2inx[line[1]]))
            else:
                tag2inx[line[1]] = len(tag2inx)
                inx2tag[len(inx2tag)] = line[1]
                sen_y.append(tag2inx[line[1]])
        num_labels=len(tag2inx)
        all_data['x']=train_x
        all_data['y']=train_y
        all_data['mask']=train_mask
        all_data['num_labels']=num_labels
        all_data['inx2tag']=inx2tag
        all_data['tag2inx']=tag2inx
    import torch
    torch.save(all_data,output_path)


    return all_data
if __name__=='__main__':
    dataPath='./ner_data/train_data'
    max_len=64

    train_data=get_data(dataPath,max_len,'./ner_data/train_data_dic')
    test_data=get_data('./ner_data/test_data',max_len,'./ner_data/test_data_dic')