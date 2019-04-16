"""
@autor: chenzf
@file: download_squad_data.py
@time: 2019/4/9 11:55 AM

"""
from urllib import request

import os
os.system("curl -d 'DDDDD=2017140433&upass=215035&0MKKey=''' '10.3.8.211'")
if not os.path.exists('./SQUAD_Data/dev-v1.1.json'):
    print('write  ./SQUAD_Data/dev-v1.1.json')
    with request.urlopen('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json') as web:
        with open('./SQUAD_Data/dev-v1.1.json','w') as f:
            f.write(web.read().decode("utf-8"))


if not os.path.exists('./SQUAD_Data/train-v1.1.json'):
    print('write  ./SQUAD_Data/train-v1.1.json')
    with request.urlopen('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json') as web:
        with open('./SQUAD_Data/train-v1.1.json','w') as f:
            f.write(web.read().decode('utf-8'))