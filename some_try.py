"""
@autor: chenzf
@file: some_try.py
@time: 2019/4/14 6:28 PM

"""
# from bert_serving.client import BertClient
# bc = BertClient()
# res=bc.encode(['你好', '你 好', '你好 啊','中国','中国第一'])
# pass

from nltk.tokenize import word_tokenize,sent_tokenize

print(sent_tokenize('you are beautiful. i think you are good.'))